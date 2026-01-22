import logging
import os
import queue
import threading
import time
import uuid
import cachetools
from oslo_concurrency import lockutils
from oslo_utils import eventletutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging import MessageDeliveryFailure
class ReplyWaiter(object):

    def __init__(self, reply_q, conn, allowed_remote_exmods):
        self.conn = conn
        self.allowed_remote_exmods = allowed_remote_exmods
        self.msg_id_cache = rpc_amqp._MsgIdCache()
        self.waiters = ReplyWaiters()
        self.conn.declare_direct_consumer(reply_q, self)
        self._thread_exit_event = eventletutils.Event()
        self._thread = threading.Thread(target=self.poll)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        if self._thread:
            self._thread_exit_event.set()
            self.conn.stop_consuming()
            self._thread.join()
            self._thread = None

    def poll(self):
        current_timeout = ACK_REQUEUE_EVERY_SECONDS_MIN
        while not self._thread_exit_event.is_set():
            try:
                self.conn.consume(timeout=current_timeout)
            except rpc_common.Timeout:
                current_timeout = min(current_timeout * 2, ACK_REQUEUE_EVERY_SECONDS_MAX)
            except Exception:
                LOG.exception('Failed to process incoming message, retrying..')
            else:
                current_timeout = ACK_REQUEUE_EVERY_SECONDS_MIN

    def __call__(self, message):
        message.acknowledge()
        incoming_msg_id = message.pop('_msg_id', None)
        if message.get('ending'):
            LOG.debug('received reply msg_id: %s', incoming_msg_id)
        self.waiters.put(incoming_msg_id, message)

    def listen(self, msg_id):
        self.waiters.add(msg_id)

    def unlisten(self, msg_id):
        self.waiters.remove(msg_id)

    @staticmethod
    def _raise_timeout_exception(msg_id, reply_q):
        raise oslo_messaging.MessagingTimeout('Timed out waiting for a reply %(reply_q)s to message ID %(msg_id)s.', {'msg_id': msg_id, 'reply_q': reply_q})

    def _process_reply(self, data):
        self.msg_id_cache.check_duplicate_message(data)
        if data['failure']:
            failure = data['failure']
            result = rpc_common.deserialize_remote_exception(failure, self.allowed_remote_exmods)
        else:
            result = data.get('result', None)
        ending = data.get('ending', False)
        return (result, ending)

    def wait(self, msg_id, timeout, call_monitor_timeout, reply_q):
        timer = rpc_common.DecayingTimer(duration=timeout)
        timer.start()
        if call_monitor_timeout:
            call_monitor_timer = rpc_common.DecayingTimer(duration=call_monitor_timeout)
            call_monitor_timer.start()
        else:
            call_monitor_timer = None
        final_reply = None
        ending = False
        while not ending:
            timeout = timer.check_return(self._raise_timeout_exception, msg_id, reply_q)
            if call_monitor_timer and timeout > 0:
                cm_timeout = call_monitor_timer.check_return(self._raise_timeout_exception, msg_id, reply_q)
                if cm_timeout < timeout:
                    timeout = cm_timeout
            try:
                message = self.waiters.get(msg_id, timeout=timeout)
            except queue.Empty:
                self._raise_timeout_exception(msg_id, reply_q)
            reply, ending = self._process_reply(message)
            if reply is not None:
                final_reply = reply
            elif ending is False:
                LOG.debug('Call monitor heartbeat received; renewing timeout timer')
                call_monitor_timer.restart()
        return final_reply
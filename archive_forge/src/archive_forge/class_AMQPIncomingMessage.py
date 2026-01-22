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
class AMQPIncomingMessage(base.RpcIncomingMessage):

    def __init__(self, listener, ctxt, message, unique_id, msg_id, reply_q, client_timeout, obsolete_reply_queues, message_operations_handler):
        super(AMQPIncomingMessage, self).__init__(ctxt, message)
        self.listener = listener
        self.unique_id = unique_id
        self.msg_id = msg_id
        self.reply_q = reply_q
        self.client_timeout = client_timeout
        self._obsolete_reply_queues = obsolete_reply_queues
        self._message_operations_handler = message_operations_handler
        self.stopwatch = timeutils.StopWatch()
        self.stopwatch.start()

    def _send_reply(self, conn, reply=None, failure=None, ending=True):
        if not self._obsolete_reply_queues.reply_q_valid(self.reply_q, self.msg_id):
            return
        if failure:
            failure = rpc_common.serialize_remote_exception(failure)
        msg = {'result': reply, 'failure': failure, 'ending': ending, '_msg_id': self.msg_id}
        rpc_amqp._add_unique_id(msg)
        unique_id = msg[rpc_amqp.UNIQUE_ID]
        LOG.debug('sending reply msg_id: %(msg_id)s reply queue: %(reply_q)s time elapsed: %(elapsed)ss', {'msg_id': self.msg_id, 'unique_id': unique_id, 'reply_q': self.reply_q, 'elapsed': self.stopwatch.elapsed()})
        conn.direct_send(self.reply_q, rpc_common.serialize_msg(msg))

    def reply(self, reply=None, failure=None):
        if not self.msg_id:
            return
        if not self._obsolete_reply_queues.reply_q_valid(self.reply_q, self.msg_id):
            return
        duration = self.listener.driver.missing_destination_retry_timeout
        timer = rpc_common.DecayingTimer(duration=duration)
        timer.start()
        while True:
            try:
                with self.listener.driver._get_connection(rpc_common.PURPOSE_SEND) as conn:
                    self._send_reply(conn, reply, failure)
                return
            except oslo_messaging.MessageUndeliverable:
                if timer.check_return() <= 0:
                    self._obsolete_reply_queues.add(self.reply_q, self.msg_id)
                    LOG.error('The reply %(msg_id)s failed to send after %(duration)d seconds due to a missing queue (%(reply_q)s). Abandoning...', {'msg_id': self.msg_id, 'duration': duration, 'reply_q': self.reply_q})
                    return
                LOG.debug('The reply %(msg_id)s could not be sent due to a missing queue (%(reply_q)s). Retrying...', {'msg_id': self.msg_id, 'reply_q': self.reply_q})
                time.sleep(0.25)
            except rpc_amqp.AMQPDestinationNotFound as exc:
                if timer.check_return() <= 0:
                    self._obsolete_reply_queues.add(self.reply_q, self.msg_id)
                    LOG.error('The reply %(msg_id)s failed to send after %(duration)d seconds due to a broker issue (%(exc)s). Abandoning...', {'msg_id': self.msg_id, 'duration': duration, 'exc': exc})
                    return
                LOG.debug('The reply %(msg_id)s could not be sent due to a broker issue (%(exc)s). Retrying...', {'msg_id': self.msg_id, 'exc': exc})
                time.sleep(0.25)

    def heartbeat(self):
        with self.listener.driver._get_connection(rpc_common.PURPOSE_SEND) as conn:
            try:
                self._send_reply(conn, None, None, ending=False)
            except oslo_messaging.MessageUndeliverable:
                raise MessageDeliveryFailure('Heartbeat send failed. Missing queue')
            except rpc_amqp.AMQPDestinationNotFound:
                raise MessageDeliveryFailure('Heartbeat send failed. Missing exchange')

    def acknowledge(self):
        pass

    def requeue(self):
        pass
import collections
import logging
import os
import threading
import uuid
import warnings
from debtcollector import removals
from oslo_config import cfg
from oslo_messaging.target import Target
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_messaging._drivers.amqp1_driver.eventloop import compute_timeout
from oslo_messaging._drivers.amqp1_driver import opts
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common
@removals.removed_class('ProtonIncomingMessage')
class ProtonIncomingMessage(base.RpcIncomingMessage):

    def __init__(self, listener, message, disposition):
        request, ctxt, client_timeout = unmarshal_request(message)
        super(ProtonIncomingMessage, self).__init__(ctxt, request)
        self.listener = listener
        self.client_timeout = client_timeout
        self._reply_to = message.reply_to
        self._correlation_id = message.id
        self._disposition = disposition

    def heartbeat(self):
        if not self._reply_to:
            LOG.warning('Cannot send RPC heartbeat: no reply-to provided')
            return
        msg = proton.Message()
        msg.correlation_id = self._correlation_id
        msg.ttl = self.client_timeout
        task = controller.SendTask('RPC KeepAlive', msg, self._reply_to, deadline=None, retry=0, wait_for_ack=False)
        self.listener.driver._ctrl.add_task(task)
        task.wait()

    def reply(self, reply=None, failure=None):
        """Schedule an RPCReplyTask to send the reply."""
        if self._reply_to:
            response = marshal_response(reply, failure)
            response.correlation_id = self._correlation_id
            driver = self.listener.driver
            deadline = compute_timeout(driver._default_reply_timeout)
            ack = not driver._pre_settle_reply
            task = controller.SendTask('RPC Reply', response, self._reply_to, deadline, retry=driver._default_reply_retry, wait_for_ack=ack)
            driver._ctrl.add_task(task)
            rc = task.wait()
            if rc:
                LOG.debug('RPC Reply failed to send: %s', str(rc))
        else:
            LOG.debug('Ignoring reply as no reply address available')

    def acknowledge(self):
        """Schedule a MessageDispositionTask to send the settlement."""
        task = controller.MessageDispositionTask(self._disposition, released=False)
        self.listener.driver._ctrl.add_task(task)

    def requeue(self):
        """Schedule a MessageDispositionTask to release the message"""
        task = controller.MessageDispositionTask(self._disposition, released=True)
        self.listener.driver._ctrl.add_task(task)
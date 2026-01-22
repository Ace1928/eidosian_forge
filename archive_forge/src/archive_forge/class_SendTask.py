import abc
import collections
import logging
import os
import platform
import queue
import random
import sys
import threading
import time
import uuid
from oslo_utils import eventletutils
import proton
import pyngus
from oslo_messaging._drivers.amqp1_driver.addressing import AddresserFactory
from oslo_messaging._drivers.amqp1_driver.addressing import keyify
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_NOTIFY
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_RPC
from oslo_messaging._drivers.amqp1_driver import eventloop
from oslo_messaging import exceptions
from oslo_messaging.target import Target
from oslo_messaging import transport
class SendTask(Task):
    """This is the class used by the Controller to send messages to a given
    destination.
    """

    def __init__(self, name, message, target, deadline, retry, wait_for_ack, notification=False):
        super(SendTask, self).__init__()
        self.name = name
        self.target = target() if isinstance(target, Target) else target
        self.message = message
        self.deadline = deadline
        self.wait_for_ack = wait_for_ack
        self.service = SERVICE_NOTIFY if notification else SERVICE_RPC
        self.timer = None
        self._retry = None if retry is None or retry < 0 else retry
        self._wakeup = eventletutils.Event()
        self._error = None
        self._sender = None

    def wait(self):
        self._wakeup.wait()
        return self._error

    def _execute(self, controller):
        if self.deadline:
            self.timer = controller.processor.alarm(self._on_timeout, self.deadline)
        controller.send(self)

    def _prepare(self, sender):
        """Called immediately before the message is handed off to the i/o
        system.  This implies that the sender link is up.
        """
        self._sender = sender

    def _on_ack(self, state, info):
        """If wait_for_ack is True, this is called by the eventloop thread when
        the ack/nack is received from the peer.  If wait_for_ack is False this
        is called by the eventloop right after the message is written to the
        link.  In the last case state will always be set to ACCEPTED.
        """
        if state != pyngus.SenderLink.ACCEPTED:
            msg = '{name} message send to {target} failed: remote disposition: {disp}, info:{info}'.format(name=self.name, target=self.target, disp=state, info=info)
            self._error = exceptions.MessageDeliveryFailure(msg)
            LOG.warning('%s', msg)
        self._cleanup()
        self._wakeup.set()

    def _on_timeout(self):
        """Invoked by the eventloop when our timer expires
        """
        self.timer = None
        self._sender and self._sender.cancel_send(self)
        msg = '{name} message sent to {target} failed: timed out'.format(name=self.name, target=self.target)
        LOG.warning('%s', msg)
        self._error = exceptions.MessagingTimeout(msg) if self.message.ttl else exceptions.MessageDeliveryFailure(msg)
        self._cleanup()
        self._wakeup.set()

    def _on_error(self, description):
        """Invoked by the eventloop if the send operation fails for reasons
        other than timeout and nack.
        """
        msg = '{name} message sent to {target} failed: {reason}'.format(name=self.name, target=self.target, reason=description)
        LOG.warning('%s', msg)
        self._error = exceptions.MessageDeliveryFailure(msg)
        self._cleanup()
        self._wakeup.set()

    def _cleanup(self):
        self._sender = None
        if self.timer:
            self.timer.cancel()
            self.timer = None

    @property
    def _can_retry(self):
        if self._retry is not None:
            self._retry -= 1
            if self._retry < 0:
                return False
        return True
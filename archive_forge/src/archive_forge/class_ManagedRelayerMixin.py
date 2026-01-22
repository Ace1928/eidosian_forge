import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
class ManagedRelayerMixin:
    """
    SMTP Relayer which notifies a manager

    Notify the manager about successful mail, failed mail
    and broken connections
    """

    def __init__(self, manager):
        self.manager = manager

    @property
    def factory(self):
        return self._factory

    @factory.setter
    def factory(self, value):
        self._factory = value

    def sentMail(self, code, resp, numOk, addresses, log):
        """
        called when e-mail has been sent

        we will always get 0 or 1 addresses.
        """
        message = self.names[0]
        if code in smtp.SUCCESS:
            self.manager.notifySuccess(self.factory, message)
        else:
            self.manager.notifyFailure(self.factory, message)
        del self.messages[0]
        del self.names[0]

    def connectionLost(self, reason: Failure=connectionDone) -> None:
        """
        called when connection is broken

        notify manager we will try to send no more e-mail
        """
        self.manager.notifyDone(self.factory)
from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
@_machine.output()
def _notifyWaiters(self, protocol):
    """
        Notify all pending requests for a connection that a connection has been
        made.

        @param protocol: The protocol of the connection.
        @type protocol: L{IProtocol}
        """
    self._failedAttempts = 0
    self._currentConnection = protocol._protocol
    self._unawait(self._currentConnection)
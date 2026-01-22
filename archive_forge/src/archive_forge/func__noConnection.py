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
def _noConnection(self, failAfterFailures=None):
    """
        Notify the caller that no connection is expected.

        @return: L{Deferred} that is fired with L{CancelledError}.
        """
    return fail(CancelledError())
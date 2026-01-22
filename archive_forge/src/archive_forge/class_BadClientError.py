import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
class BadClientError(Exception):
    """
    Raised by BadClient at the end of every datagramReceived call to try and
    screw stuff up.
    """
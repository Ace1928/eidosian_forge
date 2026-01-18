from twisted.internet.error import ConnectionDone
from twisted.internet.protocol import Protocol
from twisted.python.failure import Failure
from twisted.trial import unittest

        C{connectionMade} and C{connectionLost} are called on the protocol by
        the C{SerialPort}.
        
from struct import calcsize, pack, unpack
from twisted.protocols.stateful import StatefulProtocol
from twisted.protocols.test import test_basic
from twisted.trial.unittest import TestCase

        Send an int32-prefixed string to the other end of the connection.
        
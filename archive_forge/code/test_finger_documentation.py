from twisted.internet.testing import StringTransport
from twisted.protocols import finger
from twisted.trial import unittest

        When L{finger.Finger} receives a blank line, it responds with a message
        rejecting the request for all online users.
        
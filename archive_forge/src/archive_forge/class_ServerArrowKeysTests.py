import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
class ServerArrowKeysTests(ByteGroupingsMixin, unittest.TestCase):
    protocolFactory = ServerProtocol
    TEST_BYTES = b'\x1b[A\x1b[B\x1b[C\x1b[D'

    def verifyResults(self, transport, proto, parser):
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for arrow in (parser.UP_ARROW, parser.DOWN_ARROW, parser.RIGHT_ARROW, parser.LEFT_ARROW):
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (arrow, None))
            self.assertEqual(occurrences(result), [])
        self.assertFalse(occurrences(proto))
import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
class ClientCursorMovementTests(ByteGroupingsMixin, unittest.TestCase):
    protocolFactory = ClientProtocol
    d2 = b'\x1b[2B'
    r4 = b'\x1b[4C'
    u1 = b'\x1b[A'
    l2 = b'\x1b[2D'
    TEST_BYTES = d2 + r4 + u1 + l2 + u1 + l2
    del d2, r4, u1, l2

    def verifyResults(self, transport, proto, parser):
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for method, count in [('Down', 2), ('Forward', 4), ('Up', 1), ('Backward', 2), ('Up', 1), ('Backward', 2)]:
            result = self.assertCall(occurrences(proto).pop(0), 'cursor' + method, (count,))
            self.assertEqual(occurrences(result), [])
        self.assertFalse(occurrences(proto))
import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
class ByteGroupingsMixin(MockMixin):
    protocolFactory: Optional[Type[Protocol]] = None
    for word, n in [('Pairs', 2), ('Triples', 3), ('Quads', 4), ('Quints', 5), ('Sexes', 6)]:
        exec(_byteGroupingTestTemplate % {'groupName': word, 'bytesPer': n})
    del word, n

    def verifyResults(self, transport, proto, parser):
        result = self.assertCall(occurrences(proto).pop(0), 'makeConnection', (parser,))
        self.assertEqual(occurrences(result), [])
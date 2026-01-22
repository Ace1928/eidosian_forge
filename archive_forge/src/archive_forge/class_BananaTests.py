import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
class BananaTests(BananaTestBase):
    """
    General banana tests.
    """

    def test_string(self):
        self.enc.sendEncoded(b'hello')
        self.enc.dataReceived(self.io.getvalue())
        assert self.result == b'hello'

    def test_unsupportedUnicode(self):
        """
        Banana does not support unicode.  ``Banana.sendEncoded`` raises
        ``BananaError`` if called with an instance of ``unicode``.
        """
        self._unsupportedTypeTest('hello', 'builtins.str')

    def test_unsupportedBuiltinType(self):
        """
        Banana does not support arbitrary builtin types like L{type}.
        L{banana.Banana.sendEncoded} raises L{banana.BananaError} if called
        with an instance of L{type}.
        """
        self._unsupportedTypeTest(type, 'builtins.type')

    def test_unsupportedUserType(self):
        """
        Banana does not support arbitrary user-defined types (such as those
        defined with the ``class`` statement).  ``Banana.sendEncoded`` raises
        ``BananaError`` if called with an instance of such a type.
        """
        self._unsupportedTypeTest(MathTests(), __name__ + '.MathTests')

    def _unsupportedTypeTest(self, obj, name):
        """
        Assert that L{banana.Banana.sendEncoded} raises L{banana.BananaError}
        if called with the given object.

        @param obj: Some object that Banana does not support.
        @param name: The name of the type of the object.

        @raise: The failure exception is raised if L{Banana.sendEncoded} does
            not raise L{banana.BananaError} or if the message associated with the
            exception is not formatted to include the type of the unsupported
            object.
        """
        exc = self.assertRaises(banana.BananaError, self.enc.sendEncoded, obj)
        self.assertIn(f'Banana cannot send {name} objects', str(exc))

    def test_int(self):
        """
        A positive integer less than 2 ** 32 should round-trip through
        banana without changing value and should come out represented
        as an C{int} (regardless of the type which was encoded).
        """
        self.enc.sendEncoded(10151)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, 10151)
        self.assertIsInstance(self.result, int)

    def _getSmallest(self):
        bytes = self.enc.prefixLimit
        bits = bytes * 7
        largest = 2 ** bits - 1
        smallest = largest + 1
        return smallest

    def test_encodeTooLargeLong(self):
        """
        Test that a long above the implementation-specific limit is rejected
        as too large to be encoded.
        """
        smallest = self._getSmallest()
        self.assertRaises(banana.BananaError, self.enc.sendEncoded, smallest)

    def test_decodeTooLargeLong(self):
        """
        Test that a long above the implementation specific limit is rejected
        as too large to be decoded.
        """
        smallest = self._getSmallest()
        self.enc.setPrefixLimit(self.enc.prefixLimit * 2)
        self.enc.sendEncoded(smallest)
        encoded = self.io.getvalue()
        self.io.truncate(0)
        self.enc.setPrefixLimit(self.enc.prefixLimit // 2)
        self.assertRaises(banana.BananaError, self.enc.dataReceived, encoded)

    def _getLargest(self):
        return -self._getSmallest()

    def test_encodeTooSmallLong(self):
        """
        Test that a negative long below the implementation-specific limit is
        rejected as too small to be encoded.
        """
        largest = self._getLargest()
        self.assertRaises(banana.BananaError, self.enc.sendEncoded, largest)

    def test_decodeTooSmallLong(self):
        """
        Test that a negative long below the implementation specific limit is
        rejected as too small to be decoded.
        """
        largest = self._getLargest()
        self.enc.setPrefixLimit(self.enc.prefixLimit * 2)
        self.enc.sendEncoded(largest)
        encoded = self.io.getvalue()
        self.io.truncate(0)
        self.enc.setPrefixLimit(self.enc.prefixLimit // 2)
        self.assertRaises(banana.BananaError, self.enc.dataReceived, encoded)

    def test_integer(self):
        self.enc.sendEncoded(1015)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, 1015)

    def test_negative(self):
        self.enc.sendEncoded(-1015)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, -1015)

    def test_float(self):
        self.enc.sendEncoded(1015.0)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, 1015.0)

    def test_list(self):
        foo = [1, 2, [3, 4], [30.5, 40.2], 5, [b'six', b'seven', [b'eight', 9]], [10], []]
        self.enc.sendEncoded(foo)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, foo)

    def test_partial(self):
        """
        Test feeding the data byte per byte to the receiver. Normally
        data is not split.
        """
        foo = [1, 2, [3, 4], [30.5, 40.2], 5, [b'six', b'seven', [b'eight', 9]], [10], sys.maxsize * 3, sys.maxsize * 2, sys.maxsize * -2]
        self.enc.sendEncoded(foo)
        self.feed(self.io.getvalue())
        self.assertEqual(self.result, foo)

    def feed(self, data):
        """
        Feed the data byte per byte to the receiver.

        @param data: The bytes to deliver.
        @type data: L{bytes}
        """
        for byte in iterbytes(data):
            self.enc.dataReceived(byte)

    def test_oversizedList(self):
        data = b'\x02\x01\x01\x01\x01\x80'
        self.assertRaises(banana.BananaError, self.feed, data)

    def test_oversizedString(self):
        data = b'\x02\x01\x01\x01\x01\x82'
        self.assertRaises(banana.BananaError, self.feed, data)

    def test_crashString(self):
        crashString = b'\x00\x00\x00\x00\x04\x80'
        try:
            self.enc.dataReceived(crashString)
        except banana.BananaError:
            pass

    def test_crashNegativeLong(self):
        self.enc.sendEncoded(-2147483648)
        self.enc.dataReceived(self.io.getvalue())
        self.assertEqual(self.result, -2147483648)

    def test_sizedIntegerTypes(self):
        """
        Test that integers below the maximum C{INT} token size cutoff are
        serialized as C{INT} or C{NEG} and that larger integers are
        serialized as C{LONGINT} or C{LONGNEG}.
        """
        baseIntIn = +2147483647
        baseNegIn = -2147483648
        baseIntOut = b'\x7f\x7f\x7f\x07\x81'
        self.assertEqual(self.encode(baseIntIn - 2), b'}' + baseIntOut)
        self.assertEqual(self.encode(baseIntIn - 1), b'~' + baseIntOut)
        self.assertEqual(self.encode(baseIntIn - 0), b'\x7f' + baseIntOut)
        baseLongIntOut = b'\x00\x00\x00\x08\x85'
        self.assertEqual(self.encode(baseIntIn + 1), b'\x00' + baseLongIntOut)
        self.assertEqual(self.encode(baseIntIn + 2), b'\x01' + baseLongIntOut)
        self.assertEqual(self.encode(baseIntIn + 3), b'\x02' + baseLongIntOut)
        baseNegOut = b'\x7f\x7f\x7f\x07\x83'
        self.assertEqual(self.encode(baseNegIn + 2), b'~' + baseNegOut)
        self.assertEqual(self.encode(baseNegIn + 1), b'\x7f' + baseNegOut)
        self.assertEqual(self.encode(baseNegIn + 0), b'\x00\x00\x00\x00\x08\x83')
        baseLongNegOut = b'\x00\x00\x00\x08\x86'
        self.assertEqual(self.encode(baseNegIn - 1), b'\x01' + baseLongNegOut)
        self.assertEqual(self.encode(baseNegIn - 2), b'\x02' + baseLongNegOut)
        self.assertEqual(self.encode(baseNegIn - 3), b'\x03' + baseLongNegOut)
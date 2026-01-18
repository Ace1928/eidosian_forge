import unittest
from jsbeautifier.unpackers.urlencode import detect, unpack
def test_unpack(self):
    """Test unpack function."""

    def equals(source, result):
        return self.assertEqual(unpack(source), result)
    equals('', '')
    equals('abcd', 'abcd')
    equals('var a = b', 'var a = b')
    equals('var%20a=b', 'var a=b')
    equals('var%20a+=+b', 'var a = b')
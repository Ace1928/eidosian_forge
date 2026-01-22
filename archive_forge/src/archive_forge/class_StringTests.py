import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class StringTests(SynchronousTestCase):
    """
    Compatibility functions and types for strings.
    """

    def assertNativeString(self, original, expected):
        """
        Raise an exception indicating a failed test if the output of
        C{nativeString(original)} is unequal to the expected string, or is not
        a native string.
        """
        self.assertEqual(nativeString(original), expected)
        self.assertIsInstance(nativeString(original), str)

    def test_nonASCIIBytesToString(self):
        """
        C{nativeString} raises a C{UnicodeError} if input bytes are not ASCII
        decodable.
        """
        self.assertRaises(UnicodeError, nativeString, b'\xff')

    def test_nonASCIIUnicodeToString(self):
        """
        C{nativeString} raises a C{UnicodeError} if input Unicode is not ASCII
        encodable.
        """
        self.assertRaises(UnicodeError, nativeString, 'ሴ')

    def test_bytesToString(self):
        """
        C{nativeString} converts bytes to the native string format, assuming
        an ASCII encoding if applicable.
        """
        self.assertNativeString(b'hello', 'hello')

    def test_unicodeToString(self):
        """
        C{nativeString} converts unicode to the native string format, assuming
        an ASCII encoding if applicable.
        """
        self.assertNativeString('Good day', 'Good day')

    def test_stringToString(self):
        """
        C{nativeString} leaves native strings as native strings.
        """
        self.assertNativeString('Hello!', 'Hello!')

    def test_unexpectedType(self):
        """
        C{nativeString} raises a C{TypeError} if given an object that is not a
        string of some sort.
        """
        self.assertRaises(TypeError, nativeString, 1)
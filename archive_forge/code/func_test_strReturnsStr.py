from twisted.python import urlpath
from twisted.trial import unittest
def test_strReturnsStr(self):
    """
        Calling C{str()} with a L{URLPath} will always return a L{str}.
        """
    self.assertEqual(type(self.path.__str__()), str)
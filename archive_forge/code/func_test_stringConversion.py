from twisted.python import urlpath
from twisted.trial import unittest
def test_stringConversion(self):
    """
        Calling C{str()} with a L{URLPath} will return the same URL that it was
        constructed with.
        """
    self.assertEqual(str(self.path), 'http://example.com/foo/bar?yes=no&no=yes#footer')
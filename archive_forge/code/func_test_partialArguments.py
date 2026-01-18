from twisted.python import urlpath
from twisted.trial import unittest
def test_partialArguments(self):
    """
        Leaving some optional arguments unfilled makes a L{URLPath} with those
        optional arguments filled with defaults.
        """
    url = urlpath.URLPath.fromBytes(b'http://google.com')
    self.assertEqual(url.scheme, b'http')
    self.assertEqual(url.netloc, b'google.com')
    self.assertEqual(url.path, b'/')
    self.assertEqual(url.fragment, b'')
    self.assertEqual(url.query, b'')
    self.assertEqual(str(url), 'http://google.com/')
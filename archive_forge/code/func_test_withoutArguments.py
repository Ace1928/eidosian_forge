from twisted.python import urlpath
from twisted.trial import unittest
def test_withoutArguments(self):
    """
        An instantiation with no arguments creates a usable L{URLPath} with
        default arguments.
        """
    url = urlpath.URLPath()
    self.assertEqual(str(url), 'http://localhost/')
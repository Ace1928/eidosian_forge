from twisted.python import urlpath
from twisted.trial import unittest
def test_siblingString(self):
    """
        Calling C{str()} with a C{URLPath.sibling()} will return a URL which is
        the sibling of the URL it was instantiated with.
        """
    self.assertEqual(str(self.path.sibling(b'baz')), 'http://example.com/foo/baz')
    self.assertEqual(str(self.path.sibling(b'baz', keepQuery=True)), 'http://example.com/foo/baz?yes=no&no=yes')
    self.assertEqual(str(self.path.child(b'').sibling(b'baz')), 'http://example.com/foo/bar/baz')
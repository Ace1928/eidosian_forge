from twisted.python import urlpath
from twisted.trial import unittest
def test_parentString(self):
    """
        Calling C{str()} with a C{URLPath.parent()} will return a URL which is
        the parent of the URL it was instantiated with.
        """
    self.assertEqual(str(self.path.parent()), 'http://example.com/')
    self.assertEqual(str(self.path.parent(keepQuery=True)), 'http://example.com/?yes=no&no=yes')
    self.assertEqual(str(self.path.child(b'').parent()), 'http://example.com/foo/')
    self.assertEqual(str(self.path.child(b'baz').parent()), 'http://example.com/foo/')
    self.assertEqual(str(self.path.parent().parent().parent().parent().parent()), 'http://example.com/')
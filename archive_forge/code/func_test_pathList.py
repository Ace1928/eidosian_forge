from twisted.python import urlpath
from twisted.trial import unittest
def test_pathList(self):
    """
        L{urlpath.URLPath.pathList} returns a L{list} of L{bytes}.
        """
    self.assertEqual(self.path.child(b'%00%01%02').pathList(), [b'', b'foo', b'bar', b'%00%01%02'])
    self.assertEqual(self.path.child(b'%00%01%02').pathList(copy=False), [b'', b'foo', b'bar', b'%00%01%02'])
    self.assertEqual(self.path.child(b'%00%01%02').pathList(unquote=True), [b'', b'foo', b'bar', b'\x00\x01\x02'])
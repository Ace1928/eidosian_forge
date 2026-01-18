from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_preserveFragments(self):
    """
        L{client._urljoin} preserves the fragment identifier from either the
        new path or the base URL respectively, as specified in the HTTP 1.1 bis
        draft.

        @see: U{https://tools.ietf.org/html/draft-ietf-httpbis-p2-semantics-22#section-7.1.2}
        """
    self.assertEqual(client._urljoin(b'http://foo.com/bar#frag', b'/quux'), b'http://foo.com/quux#frag')
    self.assertEqual(client._urljoin(b'http://foo.com/bar', b'/quux#frag2'), b'http://foo.com/quux#frag2')
    self.assertEqual(client._urljoin(b'http://foo.com/bar#frag', b'/quux#frag2'), b'http://foo.com/quux#frag2')
from ...revision import Revision
from ..chk_serializer import chk_bencode_serializer
from . import TestCase
def test_roundtrips_xml_invalid_chars(self):
    rev = Revision(b'revid1')
    rev.message = '\t\ue000'
    rev.committer = 'Erik Bågfors'
    rev.timestamp = 1242385452
    rev.timezone = 3600
    rev.inventory_sha1 = b'4a2c7fb50e077699242cf6eb16a61779c7b680a7'
    self.assertRoundTrips(chk_bencode_serializer, rev)
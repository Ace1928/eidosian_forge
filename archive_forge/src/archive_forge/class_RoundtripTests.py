from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
class RoundtripTests(TestCase):

    def test_revid(self):
        md = parse_roundtripping_metadata(b'revision-id: foo\n')
        self.assertEqual(b'foo', md.revision_id)

    def test_parent_ids(self):
        md = parse_roundtripping_metadata(b'parent-ids: foo bar\n')
        self.assertEqual((b'foo', b'bar'), md.explicit_parent_ids)

    def test_properties(self):
        md = parse_roundtripping_metadata(b'property-foop: blar\n')
        self.assertEqual({b'foop': b'blar'}, md.properties)
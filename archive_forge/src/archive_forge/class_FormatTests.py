from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
class FormatTests(TestCase):

    def test_revid(self):
        metadata = CommitSupplement()
        metadata.revision_id = b'bla'
        self.assertEqual(b'revision-id: bla\n', generate_roundtripping_metadata(metadata, 'utf-8'))

    def test_parent_ids(self):
        metadata = CommitSupplement()
        metadata.explicit_parent_ids = (b'foo', b'bar')
        self.assertEqual(b'parent-ids: foo bar\n', generate_roundtripping_metadata(metadata, 'utf-8'))

    def test_properties(self):
        metadata = CommitSupplement()
        metadata.properties = {b'foo': b'bar'}
        self.assertEqual(b'property-foo: bar\n', generate_roundtripping_metadata(metadata, 'utf-8'))

    def test_empty(self):
        metadata = CommitSupplement()
        self.assertEqual(b'', generate_roundtripping_metadata(metadata, 'utf-8'))
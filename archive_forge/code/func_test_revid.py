from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
def test_revid(self):
    metadata = CommitSupplement()
    metadata.revision_id = b'bla'
    self.assertEqual(b'revision-id: bla\n', generate_roundtripping_metadata(metadata, 'utf-8'))
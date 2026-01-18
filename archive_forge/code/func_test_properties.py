from ...tests import TestCase
from ..roundtrip import (CommitSupplement, extract_bzr_metadata,
def test_properties(self):
    metadata = CommitSupplement()
    metadata.properties = {b'foo': b'bar'}
    self.assertEqual(b'property-foo: bar\n', generate_roundtripping_metadata(metadata, 'utf-8'))
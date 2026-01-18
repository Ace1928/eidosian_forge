from ...revision import Revision
from ..chk_serializer import chk_bencode_serializer
from . import TestCase
def test_written_form_matches(self):
    rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1)
    as_str = chk_bencode_serializer.write_revision_to_string(rev)
    self.assertEqualDiff(_working_revision_bencode1, as_str)
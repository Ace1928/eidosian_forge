from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_pack_revision_5(self):
    """Pack revision to XML v5"""
    rev = breezy.bzr.xml5.serializer_v5.read_revision_from_string(_revision_v5)
    outfile_contents = breezy.bzr.xml5.serializer_v5.write_revision_to_string(rev)
    self.assertEqual(outfile_contents[-1:], b'\n')
    self.assertEqualDiff(outfile_contents, b''.join(breezy.bzr.xml5.serializer_v5.write_revision_to_lines(rev)))
    self.assertEqualDiff(outfile_contents, _expected_rev_v5)
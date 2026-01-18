from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_unpack_revision_5_utc(self):
    inp = BytesIO(_revision_v5_utc)
    rev = breezy.bzr.xml5.serializer_v5.read_revision(inp)
    eq = self.assertEqual
    eq(rev.committer, 'Martin Pool <mbp@sourcefrog.net>')
    eq(len(rev.parent_ids), 1)
    eq(rev.timezone, 0)
    eq(rev.parent_ids[0], b'mbp@sourcefrog.net-20050905063503-43948f59fa127d92')
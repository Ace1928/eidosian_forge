from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_repack_revision_5(self):
    """Round-trip revision to XML v5"""
    self.check_repack_revision(_revision_v5)
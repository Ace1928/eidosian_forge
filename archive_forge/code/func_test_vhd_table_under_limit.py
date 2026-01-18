import io
import os
import re
import struct
import subprocess
import tempfile
from unittest import mock
from oslo_utils import units
from glance.common import format_inspector
from glance.tests import utils as test_utils
def test_vhd_table_under_limit(self):
    ins = format_inspector.VHDXInspector()
    meta = format_inspector.CaptureRegion(0, 0)
    desired = b'012345678ABCDEF0'
    meta.data = self._make_vhd_meta(desired, 16 * 2048)
    ins.new_region('metadata', meta)
    new_region = ins._find_meta_entry(ins._guid(desired))
    self.assertEqual(16 * 2048, new_region.length)
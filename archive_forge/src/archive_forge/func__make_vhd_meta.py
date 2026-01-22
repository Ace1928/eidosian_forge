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
def _make_vhd_meta(self, guid_raw, item_length):
    data = struct.pack('<8sHH', b'metadata', 0, 1)
    data += b'0' * 20
    data += guid_raw
    data += struct.pack('<III', 256, item_length, 0)
    data += b'0' * 6
    return data
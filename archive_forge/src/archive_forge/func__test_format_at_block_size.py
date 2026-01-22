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
def _test_format_at_block_size(self, format_name, img, block_size):
    fmt = format_inspector.get_inspector(format_name)()
    self.assertIsNotNone(fmt, 'Did not get format inspector for %s' % format_name)
    wrapper = format_inspector.InfoWrapper(open(img, 'rb'), fmt)
    while True:
        chunk = wrapper.read(block_size)
        if not chunk:
            break
    wrapper.close()
    return fmt
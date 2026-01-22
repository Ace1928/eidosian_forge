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
def _test_format_with_invalid_data(self, format_name):
    fmt = format_inspector.get_inspector(format_name)()
    wrapper = format_inspector.InfoWrapper(open(__file__, 'rb'), fmt)
    while True:
        chunk = wrapper.read(32)
        if not chunk:
            break
    wrapper.close()
    self.assertFalse(fmt.format_match)
    self.assertEqual(0, fmt.virtual_size)
    memory = sum(fmt.context_info.values())
    self.assertLess(memory, 512 * units.Ki, 'Format used more than 512KiB of memory: %s' % fmt.context_info)
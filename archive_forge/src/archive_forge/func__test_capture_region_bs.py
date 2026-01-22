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
def _test_capture_region_bs(self, bs):
    data = b''.join((chr(x).encode() for x in range(ord('A'), ord('z'))))
    regions = [format_inspector.CaptureRegion(3, 9), format_inspector.CaptureRegion(0, 256), format_inspector.CaptureRegion(32, 8)]
    for region in regions:
        self.assertFalse(region.complete)
    pos = 0
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        pos += len(chunk)
        for region in regions:
            region.capture(chunk, pos)
    self.assertEqual(data[3:12], regions[0].data)
    self.assertEqual(data[0:256], regions[1].data)
    self.assertEqual(data[32:40], regions[2].data)
    self.assertTrue(regions[0].complete)
    self.assertTrue(regions[2].complete)
    self.assertFalse(regions[1].complete)
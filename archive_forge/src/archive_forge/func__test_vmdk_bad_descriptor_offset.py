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
def _test_vmdk_bad_descriptor_offset(self, subformat=None):
    format_name = 'vmdk'
    image_size = 10 * units.Mi
    descriptorOffsetAddr = 28
    BAD_ADDRESS = 1024
    img = self._create_img(format_name, image_size, subformat=subformat)
    fd = open(img, 'r+b')
    fd.seek(descriptorOffsetAddr)
    fd.write(struct.pack('<Q', BAD_ADDRESS // 512))
    fd.close()
    for block_size in (64 * units.Ki, 512, 17, 1 * units.Mi):
        fmt = self._test_format_at_block_size(format_name, img, block_size)
        self.assertTrue(fmt.format_match, 'Failed to match %s at size %i block %i' % (format_name, image_size, block_size))
        self.assertEqual(0, fmt.virtual_size, 'Calculated a virtual size for a corrupt %s at size %i block %i' % (format_name, image_size, block_size))
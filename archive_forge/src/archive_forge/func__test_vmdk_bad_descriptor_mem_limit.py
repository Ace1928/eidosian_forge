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
def _test_vmdk_bad_descriptor_mem_limit(self, subformat=None):
    format_name = 'vmdk'
    image_size = 5 * units.Mi
    virtual_size = 5 * units.Mi
    descriptorOffsetAddr = 28
    descriptorSizeAddr = descriptorOffsetAddr + 8
    twoMBInSectors = (2 << 20) // 512
    img = self._create_allocated_vmdk(image_size // units.Mi, subformat=subformat)
    fd = open(img, 'r+b')
    fd.seek(descriptorSizeAddr)
    fd.write(struct.pack('<Q', twoMBInSectors))
    fd.close()
    for block_size in (64 * units.Ki, 512, 17, 1 * units.Mi):
        fmt = self._test_format_at_block_size(format_name, img, block_size)
        self.assertTrue(fmt.format_match, 'Failed to match %s at size %i block %i' % (format_name, image_size, block_size))
        self.assertEqual(virtual_size, fmt.virtual_size, 'Failed to calculate size for %s at size %i block %i' % (format_name, image_size, block_size))
        memory = sum(fmt.context_info.values())
        self.assertLess(memory, 1.5 * units.Mi, 'Format used more than 1.5MiB of memory: %s' % fmt.context_info)
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
class TestFormatInspectorInfra(test_utils.BaseTestCase):

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

    def test_capture_region(self):
        for block_size in (1, 3, 7, 13, 32, 64):
            self._test_capture_region_bs(block_size)

    def _get_wrapper(self, data):
        source = io.BytesIO(data)
        fake_fmt = mock.create_autospec(format_inspector.get_inspector('raw'))
        return format_inspector.InfoWrapper(source, fake_fmt)

    def test_info_wrapper_file_like(self):
        data = b''.join((chr(x).encode() for x in range(ord('A'), ord('z'))))
        wrapper = self._get_wrapper(data)
        read_data = b''
        while True:
            chunk = wrapper.read(8)
            if not chunk:
                break
            read_data += chunk
        self.assertEqual(data, read_data)

    def test_info_wrapper_iter_like(self):
        data = b''.join((chr(x).encode() for x in range(ord('A'), ord('z'))))
        wrapper = self._get_wrapper(data)
        read_data = b''
        for chunk in wrapper:
            read_data += chunk
        self.assertEqual(data, read_data)

    def test_info_wrapper_file_like_eats_error(self):
        wrapper = self._get_wrapper(b'123456')
        wrapper._format.eat_chunk.side_effect = Exception('fail')
        data = b''
        while True:
            chunk = wrapper.read(3)
            if not chunk:
                break
            data += chunk
        self.assertEqual(b'123456', data)
        wrapper._format.eat_chunk.assert_called_once_with(b'123')

    def test_info_wrapper_iter_like_eats_error(self):
        fake_fmt = mock.create_autospec(format_inspector.get_inspector('raw'))
        wrapper = format_inspector.InfoWrapper(iter([b'123', b'456']), fake_fmt)
        fake_fmt.eat_chunk.side_effect = Exception('fail')
        data = b''
        for chunk in wrapper:
            data += chunk
        self.assertEqual(b'123456', data)
        fake_fmt.eat_chunk.assert_called_once_with(b'123')

    def test_get_inspector(self):
        self.assertEqual(format_inspector.QcowInspector, format_inspector.get_inspector('qcow2'))
        self.assertIsNone(format_inspector.get_inspector('foo'))
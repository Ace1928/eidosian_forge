import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
@mock.patch('debtcollector.deprecate')
def test_qemu_img_info(self, mock_deprecate):
    img_output = '{\n                       "virtual-size": 41126400,\n                       "filename": "fake_img",\n                       "cluster-size": 65536,\n                       "format": "qcow2",\n                       "actual-size": 13168640,\n                       "format-specific": {"data": {"foo": "bar"}},\n                       "encrypted": true\n                      }'
    image_info = imageutils.QemuImgInfo(img_output, format='json')
    mock_deprecate.assert_not_called()
    self.assertEqual(41126400, image_info.virtual_size)
    self.assertEqual('fake_img', image_info.image)
    self.assertEqual(65536, image_info.cluster_size)
    self.assertEqual('qcow2', image_info.file_format)
    self.assertEqual(13168640, image_info.disk_size)
    self.assertEqual('bar', image_info.format_specific['data']['foo'])
    self.assertEqual('yes', image_info.encrypted)
    expected_str = "format_specific: {'data': {'foo': 'bar'}}"
    self.assertIn(expected_str, str(image_info))
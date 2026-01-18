import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
@mock.patch('debtcollector.deprecate')
def test_qemu_img_info_blank(self, mock_deprecate):
    img_output = '{}'
    image_info = imageutils.QemuImgInfo(img_output, format='json')
    mock_deprecate.assert_not_called()
    self.assertIsNone(image_info.virtual_size)
    self.assertIsNone(image_info.image)
    self.assertIsNone(image_info.cluster_size)
    self.assertIsNone(image_info.file_format)
    self.assertIsNone(image_info.disk_size)
    self.assertIsNone(image_info.format_specific)
    self.assertIsNone(image_info.encrypted)
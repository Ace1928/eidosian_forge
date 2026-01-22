import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
class ImageUtilsBlankTestCase(test_base.BaseTestCase):

    def test_qemu_img_info_blank(self):
        example_output = '\n'.join(['image: None', 'file_format: None', 'virtual_size: None', 'disk_size: None', 'cluster_size: None', 'backing_file: None', 'backing_file_format: None'])
        image_info = imageutils.QemuImgInfo()
        self.assertEqual(str(image_info), example_output)
        self.assertEqual(len(image_info.snapshots), 0)
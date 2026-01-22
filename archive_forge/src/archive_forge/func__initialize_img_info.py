import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
def _initialize_img_info(self):
    return ('image: %s' % self.image_name, 'file_format: %s' % self.file_format, 'virtual_size: %s' % self.virtual_size, 'disk_size: %s' % self.disk_size)
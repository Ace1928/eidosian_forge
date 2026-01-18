from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.images.ImageManager.list')
def test_zun_image_list_success(self, mock_list):
    self._test_arg_success('image-list')
    self.assertTrue(mock_list.called)
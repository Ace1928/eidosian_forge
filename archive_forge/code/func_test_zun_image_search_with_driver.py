from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.images.ImageManager.search_image')
def test_zun_image_search_with_driver(self, mock_search_image):
    self._test_arg_success('image-search 111 --image_driver glance')
    self.assertTrue(mock_search_image.called)
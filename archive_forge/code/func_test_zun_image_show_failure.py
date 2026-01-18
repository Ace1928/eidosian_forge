from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.images.ImageManager.get')
def test_zun_image_show_failure(self, mock_get):
    self._test_arg_failure('image-show --wrong 1111', self._unrecognized_arg_error)
    self.assertFalse(mock_get.called)
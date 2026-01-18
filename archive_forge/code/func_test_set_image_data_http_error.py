from unittest import mock
import urllib
import glance.common.exception as exception
from glance.common.scripts.image_import import main as image_import_script
from glance.common.scripts import utils
from glance.common import store_utils
import glance.tests.utils as test_utils
def test_set_image_data_http_error(self):
    uri = 'blahhttp://www.example.com'
    image = mock.Mock()
    self.assertRaises(urllib.error.URLError, image_import_script.set_image_data, image, uri, None)
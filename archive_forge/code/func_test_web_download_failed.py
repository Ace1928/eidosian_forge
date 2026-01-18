from unittest import mock
from glance_store._drivers import filesystem
from oslo_config import cfg
from glance.async_.flows._internal_plugins import web_download
from glance.async_.flows import api_image_import
import glance.common.exception
import glance.common.scripts.utils as script_utils
from glance import domain
import glance.tests.utils as test_utils
def test_web_download_failed(self):
    with mock.patch.object(script_utils, 'get_image_data_iter') as mock_iter:
        mock_iter.side_effect = glance.common.exception.NotFound
        self.assertRaises(glance.common.exception.NotFound, self.web_download_task.execute)
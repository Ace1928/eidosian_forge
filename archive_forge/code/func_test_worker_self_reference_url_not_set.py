from unittest import mock
from glance.api import common
from glance.api.v2 import cached_images
import glance.async_
from glance.common import exception
from glance.common import wsgi_app
from glance import sqlite_migration
from glance.tests import utils as test_utils
@mock.patch('glance.common.wsgi_app._get_config_files')
@mock.patch('glance.async_._THREADPOOL_MODEL', new=None)
def test_worker_self_reference_url_not_set(self, mock_conf):
    self.config(flavor='keystone+cache', group='paste_deploy')
    self.config(image_cache_driver='centralized_db')
    mock_conf.return_value = []
    self.assertRaises(RuntimeError, wsgi_app.init_app)
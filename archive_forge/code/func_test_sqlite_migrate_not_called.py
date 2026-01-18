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
@mock.patch('glance.common.config.load_paste_app', new=mock.MagicMock())
@mock.patch('glance.sqlite_migration.Migrate.migrate')
def test_sqlite_migrate_not_called(self, mock_migrate, mock_conf):
    self.config(flavor='keystone+cache', group='paste_deploy')
    self.config(image_cache_driver='sqlite')
    self.config(worker_self_reference_url='http://workerx')
    mock_conf.return_value = []
    wsgi_app.init_app()
    self.assertEqual(0, mock_migrate.call_count)
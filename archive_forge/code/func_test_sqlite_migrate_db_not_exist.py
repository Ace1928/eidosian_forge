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
@mock.patch('os.path.exists')
@mock.patch('os.path.join', new=mock.MagicMock())
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
@mock.patch('glance.sqlite_migration.Migrate.migrate')
def test_sqlite_migrate_db_not_exist(self, mock_migrate, mock_can_migrate, mock_exists, mock_conf):
    self.config(flavor='keystone+cache', group='paste_deploy')
    self.config(image_cache_driver='centralized_db')
    self.config(worker_self_reference_url='http://workerx')
    mock_can_migrate.return_value = True
    mock_exists.return_value = False
    mock_conf.return_value = []
    with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
        wsgi_app.init_app()
        mock_log.debug.assert_called_once_with('SQLite caching database not located, skipping migration')
        self.assertEqual(0, mock_migrate.call_count)
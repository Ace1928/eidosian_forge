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
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
def test_staging_store_path_check(self, mock_migrate_db, mock_exists, mock_conf):
    mock_migrate_db.return_value = False
    mock_exists.return_value = False
    mock_conf.return_value = []
    with mock.patch.object(wsgi_app, 'LOG') as mock_log:
        wsgi_app.init_app()
        mock_log.warning.assert_called_once_with('Import methods are enabled but staging directory %(path)s does not exist; Imports will fail!', {'path': '/tmp/staging/'})
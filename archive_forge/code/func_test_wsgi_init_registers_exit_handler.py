from unittest import mock
from glance.api import common
from glance.api.v2 import cached_images
import glance.async_
from glance.common import exception
from glance.common import wsgi_app
from glance import sqlite_migration
from glance.tests import utils as test_utils
@mock.patch('atexit.register')
@mock.patch('glance.common.config.load_paste_app')
@mock.patch('glance.async_.set_threadpool_model')
@mock.patch('glance.common.wsgi_app._get_config_files')
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
def test_wsgi_init_registers_exit_handler(self, mock_migrate_db, mock_config_files, mock_set_model, mock_load, mock_exit):
    mock_migrate_db.return_value = False
    mock_config_files.return_value = []
    wsgi_app.init_app()
    mock_exit.assert_called_once_with(wsgi_app.drain_workers)
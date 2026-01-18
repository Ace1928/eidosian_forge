from unittest import mock
from glance.api import common
from glance.api.v2 import cached_images
import glance.async_
from glance.common import exception
from glance.common import wsgi_app
from glance import sqlite_migration
from glance.tests import utils as test_utils
@mock.patch('glance.async_._THREADPOOL_MODEL', new=None)
@mock.patch('glance.common.config.load_paste_app')
@mock.patch('glance.common.wsgi_app._get_config_files')
@mock.patch('threading.Timer')
@mock.patch('glance.image_cache.prefetcher.Prefetcher')
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
def test_run_cache_prefetcher_middleware_disabled(self, mock_migrate_db, mock_prefetcher, mock_Timer, mock_conf, mock_load):
    mock_migrate_db.return_value = False
    mock_conf.return_value = []
    wsgi_app.init_app()
    mock_Timer.assert_not_called()
import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
@mock.patch.object(prefetcher, 'Prefetcher')
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
def test_number_of_workers_posix(self, mock_migrate_db, mock_prefetcher):
    """Ensure the number of workers matches num cpus limited to 8."""
    mock_migrate_db.return_value = False
    if os.name == 'nt':
        raise self.skipException('Unsupported platform.')

    def pid():
        i = 1
        while True:
            i = i + 1
            yield i
    with mock.patch.object(os, 'fork') as mock_fork:
        with mock.patch('oslo_concurrency.processutils.get_worker_count', return_value=4):
            mock_fork.side_effect = pid
            server = wsgi.Server()
            server.configure = mock.Mock()
            fake_application = 'fake-application'
            server.start(fake_application, None)
            self.assertEqual(4, len(server.children))
        with mock.patch('oslo_concurrency.processutils.get_worker_count', return_value=24):
            mock_fork.side_effect = pid
            server = wsgi.Server()
            server.configure = mock.Mock()
            fake_application = 'fake-application'
            server.start(fake_application, None)
            self.assertEqual(8, len(server.children))
        mock_fork.side_effect = pid
        server = wsgi.Server()
        server.configure = mock.Mock()
        fake_application = 'fake-application'
        server.start(fake_application, None)
        cpus = processutils.get_worker_count()
        expected_workers = cpus if cpus < 8 else 8
        self.assertEqual(expected_workers, len(server.children))
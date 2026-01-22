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
class ServerTest(test_utils.BaseTestCase):

    @mock.patch.object(prefetcher, 'Prefetcher')
    def test_create_pool(self, mock_prefetcher):
        """Ensure the wsgi thread pool is an eventlet.greenpool.GreenPool."""
        actual = wsgi.Server(threads=1).create_pool()
        self.assertIsInstance(actual, eventlet.greenpool.GreenPool)

    @mock.patch.object(prefetcher, 'Prefetcher')
    @mock.patch.object(wsgi.Server, 'configure_socket')
    def test_reserved_stores_not_allowed(self, mock_configure_socket, mock_prefetcher):
        """Ensure the reserved stores are not allowed"""
        enabled_backends = {'os_glance_file_store': 'file'}
        self.config(enabled_backends=enabled_backends)
        server = wsgi.Server(threads=1, initialize_glance_store=True)
        self.assertRaises(RuntimeError, server.configure)

    @mock.patch.object(prefetcher, 'Prefetcher')
    @mock.patch.object(wsgi.Server, 'configure_socket')
    @mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
    def test_http_keepalive(self, mock_migrate_db, mock_configure_socket, mock_prefetcher):
        mock_migrate_db.return_value = False
        self.config(http_keepalive=False)
        self.config(workers=0)
        server = wsgi.Server(threads=1)
        server.sock = 'fake_socket'
        with mock.patch.object(eventlet.wsgi, 'server') as mock_server:
            fake_application = 'fake-application'
            server.start(fake_application, 0)
            server.wait()
            mock_server.assert_called_once_with('fake_socket', fake_application, log=server._logger, debug=False, custom_pool=server.pool, keepalive=False, socket_timeout=900)

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

    @mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
    def test_invalid_staging_uri(self, mock_migrate_db):
        mock_migrate_db.return_value = False
        self.config(node_staging_uri='http://good.luck')
        server = wsgi.Server()
        with mock.patch.object(server, 'start_wsgi'):
            self.assertRaises(exception.GlanceException, server.start, 'fake-application', 34567)

    @mock.patch('os.path.exists')
    @mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
    def test_missing_staging_dir(self, mock_migrate_db, mock_exists):
        mock_migrate_db.return_value = False
        mock_exists.return_value = False
        server = wsgi.Server()
        with mock.patch.object(server, 'start_wsgi'):
            server.pool = mock.MagicMock()
            with mock.patch.object(wsgi, 'LOG') as mock_log:
                server.start('fake-application', 34567)
                mock_exists.assert_called_once_with('/tmp/staging/')
                mock_log.warning.assert_called_once_with('Import methods are enabled but staging directory %(path)s does not exist; Imports will fail!', {'path': '/tmp/staging/'})
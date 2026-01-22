import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
class BackdoorSocketPathTest(base.ServiceBaseTestCase):

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_path(self, listen_mock, spawn_mock):
        self.config(backdoor_socket='/tmp/my_special_socket')
        listen_mock.side_effect = mock.Mock()
        path = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual('/tmp/my_special_socket', path)

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_path_with_format_string(self, listen_mock, spawn_mock):
        self.config(backdoor_socket='/tmp/my_special_socket-{pid}')
        listen_mock.side_effect = mock.Mock()
        path = eventlet_backdoor.initialize_if_enabled(self.conf)
        expected_path = '/tmp/my_special_socket-{}'.format(os.getpid())
        self.assertEqual(expected_path, path)

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_path_with_broken_format_string(self, listen_mock, spawn_mock):
        broken_socket_paths = ['/tmp/my_special_socket-{}', '/tmp/my_special_socket-{broken', '/tmp/my_special_socket-{broken}']
        for socket_path in broken_socket_paths:
            self.config(backdoor_socket=socket_path)
            listen_mock.side_effect = mock.Mock()
            path = eventlet_backdoor.initialize_if_enabled(self.conf)
            self.assertEqual(socket_path, path)

    @mock.patch.object(os, 'unlink')
    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_path_already_exists(self, listen_mock, spawn_mock, unlink_mock):
        self.config(backdoor_socket='/tmp/my_special_socket')
        sock = mock.Mock()
        listen_mock.side_effect = [socket.error(errno.EADDRINUSE, ''), sock]
        path = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual('/tmp/my_special_socket', path)
        unlink_mock.assert_called_with('/tmp/my_special_socket')

    @mock.patch.object(os, 'unlink')
    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_path_already_exists_and_gone(self, listen_mock, spawn_mock, unlink_mock):
        self.config(backdoor_socket='/tmp/my_special_socket')
        sock = mock.Mock()
        listen_mock.side_effect = [socket.error(errno.EADDRINUSE, ''), sock]
        unlink_mock.side_effect = OSError(errno.ENOENT, '')
        path = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual('/tmp/my_special_socket', path)
        unlink_mock.assert_called_with('/tmp/my_special_socket')

    @mock.patch.object(os, 'unlink')
    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_path_already_exists_and_not_gone(self, listen_mock, spawn_mock, unlink_mock):
        self.config(backdoor_socket='/tmp/my_special_socket')
        listen_mock.side_effect = socket.error(errno.EADDRINUSE, '')
        unlink_mock.side_effect = OSError(errno.EPERM, '')
        self.assertRaises(OSError, eventlet_backdoor.initialize_if_enabled, self.conf)

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_path_no_perms(self, listen_mock, spawn_mock):
        self.config(backdoor_socket='/tmp/my_special_socket')
        listen_mock.side_effect = socket.error(errno.EPERM, '')
        self.assertRaises(socket.error, eventlet_backdoor.initialize_if_enabled, self.conf)
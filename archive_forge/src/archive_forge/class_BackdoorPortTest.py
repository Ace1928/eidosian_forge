import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
class BackdoorPortTest(base.ServiceBaseTestCase):

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_port(self, listen_mock, spawn_mock):
        self.config(backdoor_port=1234)
        sock = mock.Mock()
        sock.getsockname.return_value = ('127.0.0.1', 1234)
        listen_mock.return_value = sock
        port = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual(1234, port)

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_port_inuse(self, listen_mock, spawn_mock):
        self.config(backdoor_port=2345)
        listen_mock.side_effect = socket.error(errno.EADDRINUSE, '')
        self.assertRaises(socket.error, eventlet_backdoor.initialize_if_enabled, self.conf)

    @mock.patch.object(eventlet, 'spawn')
    def test_backdoor_port_range_inuse(self, spawn_mock):
        self.config(backdoor_port='8800:8801')
        port = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual(8800, port)
        port = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual(8801, port)

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_port_range(self, listen_mock, spawn_mock):
        self.config(backdoor_port='8800:8899')
        sock = mock.Mock()
        sock.getsockname.return_value = ('127.0.0.1', 8800)
        listen_mock.return_value = sock
        port = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual(8800, port)

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_port_range_one_inuse(self, listen_mock, spawn_mock):
        self.config(backdoor_port='8800:8900')
        sock = mock.Mock()
        sock.getsockname.return_value = ('127.0.0.1', 8801)
        listen_mock.side_effect = [socket.error(errno.EADDRINUSE, ''), sock]
        port = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual(8801, port)

    @mock.patch.object(eventlet, 'spawn')
    @mock.patch.object(eventlet, 'listen')
    def test_backdoor_port_range_all_inuse(self, listen_mock, spawn_mock):
        self.config(backdoor_port='8800:8899')
        side_effects = []
        for i in range(8800, 8900):
            side_effects.append(socket.error(errno.EADDRINUSE, ''))
        listen_mock.side_effect = side_effects
        self.assertRaises(socket.error, eventlet_backdoor.initialize_if_enabled, self.conf)

    def test_backdoor_port_reverse_range(self):
        self.config(backdoor_port='8888:7777')
        self.assertRaises(eventlet_backdoor.EventletBackdoorConfigValueError, eventlet_backdoor.initialize_if_enabled, self.conf)

    def test_backdoor_port_bad(self):
        self.config(backdoor_port='abc')
        self.assertRaises(eventlet_backdoor.EventletBackdoorConfigValueError, eventlet_backdoor.initialize_if_enabled, self.conf)
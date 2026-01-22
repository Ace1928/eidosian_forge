import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
class TestSocketManagerSetupAndFunctions(unittest.TestCase):
    """
    Unit/integration tests for starting up a socket.
    """

    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(1)

    def tearDown(self):
        self.fake_socket.close()

    def test_init_and_reg_shutdown(self):
        """
        Test initialization of a socket manager.
        """
        self.assertFalse(self.fake_socket.connected)
        nop_called = False

        def nop(*args):
            nonlocal nop_called
            nop_called = True
        socket_manager = SocketManager('https://127.0.0.1', self.fake_socket.port, nop, nop, nop, TASK_GROUP_ID_1, 0.3, nop)
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        socket_manager.shutdown()
        time.sleep(0.3)
        self.assertTrue(self.fake_socket.disconnected)
        self.assertTrue(socket_manager.is_shutdown)
        self.assertFalse(nop_called)

    def assertEqualBy(self, val_func, val, max_time):
        start_time = time.time()
        while val_func() != val:
            assert time.time() - start_time < max_time, 'Value was not attained in specified time, was {} rather than {}'.format(val_func(), val)
            time.sleep(0.1)

    def test_init_and_socket_shutdown(self):
        """
        Test initialization of a socket manager with a failed shutdown.
        """
        self.assertFalse(self.fake_socket.connected)
        nop_called = False

        def nop(*args):
            nonlocal nop_called
            nop_called = True
        server_death_called = False

        def server_death(*args):
            nonlocal server_death_called
            server_death_called = True
        socket_manager = SocketManager('https://127.0.0.1', self.fake_socket.port, nop, nop, nop, TASK_GROUP_ID_1, 0.4, server_death)
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        self.fake_socket.close()
        self.assertEqualBy(lambda: socket_manager.alive, False, 8)
        self.assertEqualBy(lambda: server_death_called, True, 20)
        self.assertFalse(nop_called)
        socket_manager.shutdown()

    def test_init_and_socket_shutdown_then_restart(self):
        """
        Test restoring connection to a socket.
        """
        self.assertFalse(self.fake_socket.connected)
        nop_called = False

        def nop(*args):
            nonlocal nop_called
            nop_called = True
        server_death_called = False

        def server_death(*args):
            nonlocal server_death_called
            server_death_called = True
        socket_manager = SocketManager('https://127.0.0.1', self.fake_socket.port, nop, nop, nop, TASK_GROUP_ID_1, 0.4, server_death)
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        self.fake_socket.close()
        self.assertEqualBy(lambda: socket_manager.alive, False, 8)
        self.assertFalse(socket_manager.alive)
        self.fake_socket = MockSocket()
        self.assertEqualBy(lambda: socket_manager.alive, True, 4)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)
        socket_manager.shutdown()

    def test_init_world_dead(self):
        """
        Test initialization of a socket manager with a failed startup.
        """
        self.assertFalse(self.fake_socket.connected)
        self.fake_socket.close()
        nop_called = False

        def nop(*args):
            nonlocal nop_called
            nop_called = True
        server_death_called = False

        def server_death(*args):
            nonlocal server_death_called
            server_death_called = True
        with self.assertRaises(ConnectionRefusedError):
            socket_manager = SocketManager('https://127.0.0.1', self.fake_socket.port, nop, nop, nop, TASK_GROUP_ID_1, 0.4, server_death)
            self.assertIsNone(socket_manager)
        self.assertFalse(nop_called)
        self.assertTrue(server_death_called)
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
def test_simple_packet_channel_management(self):
    """
        Ensure that channels are created, managed, and then removed as expected.
        """
    use_packet = self.MESSAGE_SEND_PACKET_1
    worker_id = use_packet.receiver_id
    assignment_id = use_packet.assignment_id
    self.socket_manager.open_channel(worker_id, assignment_id)
    time.sleep(0.1)
    connection_id = use_packet.get_receiver_connection_id()
    self.assertIn(connection_id, self.socket_manager.open_channels)
    self.assertTrue(self.socket_manager.socket_is_open(connection_id))
    self.assertFalse(self.socket_manager.socket_is_open(FAKE_ID))
    resp = self.socket_manager.queue_packet(use_packet)
    self.assertIn(use_packet.id, self.socket_manager.packet_map)
    self.assertTrue(resp)
    self.assertEqual(self.socket_manager.get_status(use_packet.id), use_packet.status)
    self.assertEqual(self.socket_manager.get_status(FAKE_ID), Packet.STATUS_NONE)
    self.socket_manager.close_channel(connection_id)
    time.sleep(0.2)
    self.assertNotIn(connection_id, self.socket_manager.open_channels)
    self.assertNotIn(use_packet.id, self.socket_manager.packet_map)
    self.socket_manager.open_channel(worker_id, assignment_id)
    self.socket_manager.open_channel(worker_id + '2', assignment_id)
    time.sleep(0.1)
    self.assertEqual(len(self.socket_manager.open_channels), 2)
    self.socket_manager.close_all_channels()
    time.sleep(0.1)
    self.assertEqual(len(self.socket_manager.open_channels), 0)
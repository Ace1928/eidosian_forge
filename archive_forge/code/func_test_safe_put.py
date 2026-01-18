import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def test_safe_put(self):
    """
        Test safe put and queue retrieval mechanisms.
        """
    self.socket_manager._send_packet = mock.MagicMock()
    use_packet = self.MESSAGE_SEND_PACKET_1
    worker_id = use_packet.receiver_id
    assignment_id = use_packet.assignment_id
    connection_id = use_packet.get_receiver_connection_id()
    self.socket_manager.open_channel(worker_id, assignment_id)
    send_time = time.time()
    self.socket_manager._safe_put(connection_id, (send_time, use_packet))
    time.sleep(0.3)
    self.socket_manager._send_packet.assert_called_once()
    call_args = self.socket_manager._send_packet.call_args[0]
    self.assertEqual(use_packet, call_args[0])
    self.assertEqual(connection_id, call_args[1])
    self.assertEqual(send_time, call_args[2])
    self.socket_manager.close_all_channels()
    time.sleep(0.1)
    self.socket_manager._safe_put(connection_id, (send_time, use_packet))
    self.assertEqual(use_packet.status, Packet.STATUS_FAIL)
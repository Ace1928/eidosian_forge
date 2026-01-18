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
def test_packet_modifications(self):
    """
        Ensure that packet copies and acts are produced properly.
        """
    self.assertEqual(self.packet_1.swap_sender(), self.packet_1)
    self.assertEqual(self.packet_1.set_type(data_model.MESSAGE_BATCH), self.packet_1)
    self.assertEqual(self.packet_1.set_data(None), self.packet_1)
    self.assertEqual(self.packet_1.sender_id, self.RECEIVER_ID)
    self.assertEqual(self.packet_1.receiver_id, self.SENDER_ID)
    self.assertEqual(self.packet_1.type, data_model.MESSAGE_BATCH)
    self.assertIsNone(self.packet_1.data)
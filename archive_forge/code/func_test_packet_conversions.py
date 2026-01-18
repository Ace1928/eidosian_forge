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
def test_packet_conversions(self):
    """
        Ensure that packet copies and acts are produced properly.
        """
    message_packet_copy = self.packet_1.new_copy()
    self.assertNotEqual(message_packet_copy.id, self.ID)
    self.assertNotEqual(message_packet_copy, self.packet_1)
    self.assertEqual(message_packet_copy.type, self.packet_1.type)
    self.assertEqual(message_packet_copy.sender_id, self.packet_1.sender_id)
    self.assertEqual(message_packet_copy.receiver_id, self.packet_1.receiver_id)
    self.assertEqual(message_packet_copy.assignment_id, self.packet_1.assignment_id)
    self.assertEqual(message_packet_copy.data, self.packet_1.data)
    self.assertEqual(message_packet_copy.conversation_id, self.packet_1.conversation_id)
    self.assertIsNone(message_packet_copy.ack_func)
    self.assertEqual(message_packet_copy.status, Packet.STATUS_INIT)
    hb_packet_copy = self.packet_2.new_copy()
    self.assertNotEqual(hb_packet_copy.id, self.ID)
    self.assertNotEqual(hb_packet_copy, self.packet_2)
    self.assertEqual(hb_packet_copy.type, self.packet_2.type)
    self.assertEqual(hb_packet_copy.sender_id, self.packet_2.sender_id)
    self.assertEqual(hb_packet_copy.receiver_id, self.packet_2.receiver_id)
    self.assertEqual(hb_packet_copy.assignment_id, self.packet_2.assignment_id)
    self.assertEqual(hb_packet_copy.data, self.packet_2.data)
    self.assertEqual(hb_packet_copy.conversation_id, self.packet_2.conversation_id)
    self.assertIsNone(hb_packet_copy.ack_func)
    self.assertEqual(hb_packet_copy.status, Packet.STATUS_INIT)
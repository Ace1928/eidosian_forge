import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.mturk.core.dev.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_send_message_command(self):
    manager = self.mturk_manager
    worker_id = self.agent_1.worker_id
    assignment_id = self.agent_1.assignment_id
    manager.socket_manager.queue_packet = mock.MagicMock()
    data = {'text': data_model.COMMAND_SEND_MESSAGE}
    manager.send_command(worker_id, assignment_id, data)
    manager.socket_manager.queue_packet.assert_called_once()
    packet = manager.socket_manager.queue_packet.call_args[0][0]
    self.assertIsNotNone(packet.id)
    self.assertEqual(packet.type, data_model.WORLD_MESSAGE)
    self.assertEqual(packet.receiver_id, worker_id)
    self.assertEqual(packet.assignment_id, assignment_id)
    self.assertEqual(packet.data, data)
    self.assertEqual(packet.data['type'], data_model.MESSAGE_TYPE_COMMAND)
    data = {'text': 'This is a test message'}
    manager.socket_manager.queue_packet.reset_mock()
    message_id = manager.send_message(worker_id, assignment_id, data)
    manager.socket_manager.queue_packet.assert_called_once()
    packet = manager.socket_manager.queue_packet.call_args[0][0]
    self.assertIsNotNone(packet.id)
    self.assertEqual(packet.type, data_model.WORLD_MESSAGE)
    self.assertEqual(packet.receiver_id, worker_id)
    self.assertEqual(packet.assignment_id, assignment_id)
    self.assertNotEqual(packet.data, data)
    self.assertEqual(data['text'], packet.data['text'])
    self.assertEqual(packet.data['message_id'], message_id)
    self.assertEqual(packet.data['type'], data_model.MESSAGE_TYPE_ACT)
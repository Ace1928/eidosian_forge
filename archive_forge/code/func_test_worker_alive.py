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
def test_worker_alive(self):
    manager = self.mturk_manager
    manager.task_group_id = 'TEST_GROUP_ID'
    manager.server_url = 'https://127.0.0.1'
    manager.task_state = manager.STATE_ACCEPTING_WORKERS
    manager._setup_socket()
    manager.force_expire_hit = mock.MagicMock()
    manager._onboard_new_agent = mock.MagicMock()
    manager.socket_manager.open_channel = mock.MagicMock(wraps=manager.socket_manager.open_channel)
    manager.worker_manager.worker_alive = mock.MagicMock(wraps=manager.worker_manager.worker_alive)
    open_channel = manager.socket_manager.open_channel
    worker_alive = manager.worker_manager.worker_alive
    alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': None, 'conversation_id': None}, '')
    manager._on_alive(alive_packet)
    open_channel.assert_not_called()
    worker_alive.assert_not_called()
    manager._onboard_new_agent.assert_not_called()
    alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_1, 'conversation_id': None}, '')
    manager.accepting_workers = False
    manager._on_alive(alive_packet)
    open_channel.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
    worker_alive.assert_called_once_with(TEST_WORKER_ID_1)
    worker_state = manager.worker_manager._get_worker(TEST_WORKER_ID_1)
    self.assertIsNotNone(worker_state)
    open_channel.reset_mock()
    worker_alive.reset_mock()
    manager.force_expire_hit.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
    manager._onboard_new_agent.assert_not_called()
    manager.force_expire_hit.reset_mock()
    manager.accepting_workers = True
    manager._on_alive(alive_packet)
    open_channel.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
    worker_alive.assert_called_once_with(TEST_WORKER_ID_1)
    manager._onboard_new_agent.assert_called_once()
    manager._onboard_new_agent.reset_mock()
    manager.force_expire_hit.assert_not_called()
    agent = manager.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
    self.assertIsInstance(agent, MTurkAgent)
    self.assertEqual(agent.get_status(), AssignState.STATUS_NONE)
    agent.set_status(AssignState.STATUS_IN_TASK)
    alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_2, 'conversation_id': None}, '')
    manager.opt['allowed_conversations'] = 1
    manager._on_alive(alive_packet)
    manager.force_expire_hit.assert_called_once()
    manager._onboard_new_agent.assert_not_called()
    manager.force_expire_hit.reset_mock()
    agent.set_status(AssignState.STATUS_DONE)
    alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_2, 'conversation_id': None}, '')
    manager.is_unique = True
    manager._on_alive(alive_packet)
    manager.force_expire_hit.assert_called_once()
    manager._onboard_new_agent.assert_not_called()
    manager.force_expire_hit.reset_mock()
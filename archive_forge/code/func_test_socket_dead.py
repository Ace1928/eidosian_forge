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
def test_socket_dead(self):
    """
        Test all states of socket dead calls.
        """
    manager = self.mturk_manager
    agent = self.agent_1
    worker_id = agent.worker_id
    assignment_id = agent.assignment_id
    manager.socket_manager.close_channel = mock.MagicMock()
    agent.reduce_state = mock.MagicMock()
    agent.set_status = mock.MagicMock(wraps=agent.set_status)
    manager._handle_agent_disconnect = mock.MagicMock(wraps=manager._handle_agent_disconnect)
    agent.set_status(AssignState.STATUS_NONE)
    agent.set_status.reset_mock()
    manager._on_socket_dead(worker_id, assignment_id)
    self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
    agent.reduce_state.assert_called_once()
    manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
    manager._handle_agent_disconnect.assert_not_called()
    agent.set_status(AssignState.STATUS_ONBOARDING)
    agent.set_status.reset_mock()
    agent.reduce_state.reset_mock()
    manager.socket_manager.close_channel.reset_mock()
    self.assertFalse(agent.disconnected)
    manager._on_socket_dead(worker_id, assignment_id)
    self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
    agent.reduce_state.assert_called_once()
    manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
    self.assertTrue(agent.disconnected)
    manager._handle_agent_disconnect.assert_not_called()
    agent.disconnected = False
    agent.set_status(AssignState.STATUS_WAITING)
    agent.set_status.reset_mock()
    agent.reduce_state.reset_mock()
    manager.socket_manager.close_channel.reset_mock()
    manager._add_agent_to_pool(agent)
    manager._remove_from_agent_pool = mock.MagicMock()
    manager._on_socket_dead(worker_id, assignment_id)
    self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
    agent.reduce_state.assert_called_once()
    manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
    self.assertTrue(agent.disconnected)
    manager._handle_agent_disconnect.assert_not_called()
    manager._remove_from_agent_pool.assert_called_once_with(agent)
    agent.disconnected = False
    agent.set_status(AssignState.STATUS_IN_TASK)
    agent.set_status.reset_mock()
    agent.reduce_state.reset_mock()
    manager.socket_manager.close_channel.reset_mock()
    manager._add_agent_to_pool(agent)
    manager._remove_from_agent_pool = mock.MagicMock()
    manager._on_socket_dead(worker_id, assignment_id)
    self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
    manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
    self.assertTrue(agent.disconnected)
    manager._handle_agent_disconnect.assert_called_once_with(worker_id, assignment_id)
    agent.disconnected = False
    agent.set_status(AssignState.STATUS_DONE)
    agent.set_status.reset_mock()
    agent.reduce_state.reset_mock()
    manager._handle_agent_disconnect.reset_mock()
    manager.socket_manager.close_channel.reset_mock()
    manager._add_agent_to_pool(agent)
    manager._remove_from_agent_pool = mock.MagicMock()
    manager._on_socket_dead(worker_id, assignment_id)
    self.assertNotEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
    agent.reduce_state.assert_not_called()
    manager.socket_manager.close_channel.assert_not_called()
    self.assertFalse(agent.disconnected)
    manager._handle_agent_disconnect.assert_not_called()
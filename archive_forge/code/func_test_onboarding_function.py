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
def test_onboarding_function(self):
    manager = self.mturk_manager
    manager.get_onboard_world = mock.MagicMock(wraps=get_onboard_world)
    manager.send_message = mock.MagicMock()
    manager._move_agents_to_waiting = mock.MagicMock()
    manager.worker_manager.get_agent_for_assignment = mock.MagicMock(return_value=self.agent_1)
    onboard_threads = manager.assignment_to_onboard_thread
    did_launch = manager._onboard_new_agent(self.agent_1)
    assert_equal_by(onboard_threads[self.agent_1.assignment_id].isAlive, True, 0.2)
    time.sleep(0.1)
    self.assertTrue(did_launch)
    manager.get_onboard_world.assert_called_with(self.agent_1)
    manager.get_onboard_world.reset_mock()
    did_launch = manager._onboard_new_agent(self.agent_1)
    manager.worker_manager.get_agent_for_assignment.assert_not_called()
    manager.get_onboard_world.assert_not_called()
    self.assertFalse(did_launch)
    assert_equal_by(onboard_threads[self.agent_1.assignment_id].isAlive, False, 3)
    manager._move_agents_to_waiting.assert_called_once()
    did_launch = manager._onboard_new_agent(self.agent_1)
    self.assertFalse(did_launch)
    self.agent_1.set_status(AssignState.STATUS_NONE)
    did_launch = manager._onboard_new_agent(self.agent_1)
    self.assertTrue(did_launch)
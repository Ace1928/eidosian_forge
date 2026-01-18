import unittest
import time
import uuid
import os
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.mturk_manager as MTurkManagerFile
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def test_return_to_waiting_on_world_start(self):
    manager = self.mturk_manager
    agent_1 = self.agent_1
    self.alive_agent(agent_1)
    assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
    agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
    self.assertFalse(self.onboarding_agents[agent_1.worker_id])
    self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
    self.onboarding_agents[agent_1.worker_id] = True
    assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

    def replace_on_msg(packet):
        agent_1.message_packet.append(packet)
    agent_1.on_msg = replace_on_msg
    agent_2 = self.agent_2
    self.alive_agent(agent_2)
    assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
    agent_2_object = manager.worker_manager.get_agent_for_assignment(agent_2.assignment_id)
    self.assertFalse(self.onboarding_agents[agent_2.worker_id])
    self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
    self.onboarding_agents[agent_2.worker_id] = True
    assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 2)
    assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
    assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 3)
    agent_1.always_beat = False
    self.assertNotIn(agent_2.worker_id, self.worlds_agents)
    manager.shutdown()
    assert_equal_by(lambda: len([x for x in manager.socket_manager.run.values() if not x]), 2, 2)
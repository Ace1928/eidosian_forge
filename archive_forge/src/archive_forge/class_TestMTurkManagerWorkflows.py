import unittest
import time
import uuid
import os
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.mturk.core.dev.agents import AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
class TestMTurkManagerWorkflows(unittest.TestCase):
    """
    Various test cases to replicate a whole mturk workflow.
    """

    def setUp(self):
        patcher = mock.patch('builtins.input', return_value='y')
        self.addCleanup(patcher.stop)
        patcher.start()
        self.server_utils = MTurkManagerFile.server_utils
        self.mturk_utils = MTurkManagerFile.mturk_utils
        self.server_utils.setup_server = mock.MagicMock(return_value='https://127.0.0.1')
        self.server_utils.setup_legacy_server = mock.MagicMock(return_value='https://127.0.0.1')
        self.server_utils.delete_server = mock.MagicMock()
        self.mturk_utils.setup_aws_credentials = mock.MagicMock()
        self.mturk_utils.calculate_mturk_cost = mock.MagicMock(return_value=1)
        self.mturk_utils.check_mturk_balance = mock.MagicMock(return_value=True)
        self.mturk_utils.create_hit_config = mock.MagicMock()
        self.mturk_utils.setup_sns_topic = mock.MagicMock(return_value=TOPIC_ARN)
        self.mturk_utils.delete_sns_topic = mock.MagicMock()
        self.mturk_utils.delete_qualification = mock.MagicMock()
        self.mturk_utils.find_or_create_qualification = mock.MagicMock(return_value=QUALIFICATION_ID)
        self.mturk_utils.find_qualification = mock.MagicMock(return_value=QUALIFICATION_ID)
        self.mturk_utils.give_worker_qualification = mock.MagicMock()
        self.mturk_utils.remove_worker_qualification = mock.MagicMock()
        self.mturk_utils.create_hit_type = mock.MagicMock(return_value=HIT_TYPE_ID)
        self.mturk_utils.subscribe_to_hits = mock.MagicMock()
        self.mturk_utils.create_hit_with_hit_type = mock.MagicMock(return_value=(MTURK_PAGE_URL, FAKE_HIT_ID, 'MTURK_HIT_DATA'))
        self.mturk_utils.get_mturk_client = mock.MagicMock(return_value=mock.MagicMock())
        self.onboarding_agents = {}
        self.worlds_agents = {}
        self.fake_socket = MockSocket()
        time.sleep(0.1)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args()
        self.opt['task'] = 'unittest'
        self.opt['frontend_version'] = 1
        self.opt['assignment_duration_in_seconds'] = 1
        self.opt['hit_title'] = 'test_hit_title'
        self.opt['hit_description'] = 'test_hit_description'
        self.opt['task_description'] = 'test_task_description'
        self.opt['hit_keywords'] = 'test_hit_keywords'
        self.opt['reward'] = 0.1
        self.opt['is_debug'] = True
        self.opt['log_level'] = 0
        self.opt['num_conversations'] = 1
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(opt=self.opt, mturk_agent_ids=self.mturk_agent_ids, is_test=True)
        self.mturk_manager.port = self.fake_socket.port
        self.mturk_manager.setup_server()
        self.mturk_manager.start_new_run()
        self.mturk_manager.ready_to_accept_workers()
        self.mturk_manager.set_get_onboard_world(self.get_onboard_world)
        self.mturk_manager.create_hits()

        def assign_worker_roles(workers):
            workers[0].id = 'mturk_agent_1'
            workers[1].id = 'mturk_agent_2'

        def run_task_wait():
            self.mturk_manager.start_task(lambda w: True, assign_worker_roles, self.get_task_world)
        self.task_thread = threading.Thread(target=run_task_wait)
        self.task_thread.start()
        self.agent_1 = MockAgent(TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1, TASK_GROUP_ID_1)
        self.agent_1_2 = MockAgent(TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_1, TASK_GROUP_ID_1)
        self.agent_2 = MockAgent(TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2, TASK_GROUP_ID_1)

    def tearDown(self):
        for key in self.worlds_agents.keys():
            self.worlds_agents[key] = True
        self.mturk_manager.shutdown()
        self.fake_socket.close()
        if self.task_thread.isAlive():
            self.task_thread.join()

    def get_onboard_world(self, mturk_agent):
        self.onboarding_agents[mturk_agent.worker_id] = False

        def episode_done():
            return not (mturk_agent.worker_id in self.onboarding_agents and self.onboarding_agents[mturk_agent.worker_id] is False)
        return TestMTurkOnboardWorld(mturk_agent, episode_done)

    def get_task_world(self, mturk_manager, opt, workers):
        for worker in workers:
            self.worlds_agents[worker.worker_id] = False

        def episode_done():
            for worker in workers:
                if self.worlds_agents[worker.worker_id] is False:
                    return False
            return True
        return TestMTurkWorld(workers, episode_done)

    def alive_agent(self, agent):
        agent.register_to_socket(self.fake_socket)
        agent.send_alive()
        time.sleep(0.3)

    def test_successful_convo(self):
        manager = self.mturk_manager
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(agent_2.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        agent_1_object.set_completed_act({})
        agent_2_object.set_completed_act({})
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)

    def test_disconnect_end(self):
        manager = self.mturk_manager
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(agent_2.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)
        agent_2.send_disconnect()
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_PARTNER_DISCONNECT, 3)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DISCONNECT, 3)
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        self.assertEqual(manager.completed_conversations, 0)
        agent_1_object.set_completed_act({})

    def test_expire_onboarding(self):
        manager = self.mturk_manager
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 10)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        manager._expire_onboarding_pool()
        self.onboarding_agents[agent_1.worker_id] = True
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_EXPIRED)

    def test_attempt_break_unique(self):
        manager = self.mturk_manager
        unique_worker_qual = 'is_unique_qual'
        manager.is_unique = True
        manager.opt['unique_qual_name'] = unique_worker_qual
        manager.unique_qual_name = unique_worker_qual
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(agent_2.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        agent_1_object.set_completed_act({})
        agent_2_object.set_completed_act({})
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)
        self.mturk_utils.find_qualification.assert_called_with(unique_worker_qual, manager.is_sandbox)
        self.mturk_utils.give_worker_qualification.assert_any_call(agent_1.worker_id, QUALIFICATION_ID, None, manager.is_sandbox)
        self.mturk_utils.give_worker_qualification.assert_any_call(agent_2.worker_id, QUALIFICATION_ID, None, manager.is_sandbox)
        agent_1_2 = self.agent_1_2
        self.alive_agent(agent_1_2)
        assert_equal_by(lambda: agent_1_2.worker_id in self.onboarding_agents, True, 2)
        agent_1_2_object = manager.worker_manager.get_agent_for_assignment(agent_1_2.assignment_id)
        self.assertIsNone(agent_1_2_object)

    def test_break_multi_convo(self):
        manager = self.mturk_manager
        manager.opt['allowed_conversations'] = 1
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(agent_2.assignment_id)
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)
        agent_1_2 = self.agent_1_2
        self.alive_agent(agent_1_2)
        assert_equal_by(lambda: agent_1_2.worker_id in self.onboarding_agents, True, 2)
        agent_1_2_object = manager.worker_manager.get_agent_for_assignment(agent_1_2.assignment_id)
        self.assertIsNone(agent_1_2_object)
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        agent_1_object.set_completed_act({})
        agent_2_object.set_completed_act({})
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)

    def test_no_onboard_expire_waiting(self):
        manager = self.mturk_manager
        manager.set_get_onboard_world(None)
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)
        manager._expire_agent_pool()
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_EXPIRED)
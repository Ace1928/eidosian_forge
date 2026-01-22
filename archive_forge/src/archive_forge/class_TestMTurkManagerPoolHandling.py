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
class TestMTurkManagerPoolHandling(unittest.TestCase):

    def setUp(self):
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args()
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(opt=self.opt, mturk_agent_ids=self.mturk_agent_ids, is_test=True)
        self.mturk_manager._init_state()
        self.agent_1 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
        self.agent_2 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2)
        self.agent_3 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_3, TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_3)

    def tearDown(self):
        self.mturk_manager.shutdown()

    def test_pool_add_get_remove_and_expire(self):
        """
        Ensure the pool properly adds and releases workers.
        """
        all_are_eligible = {'multiple': True, 'func': lambda workers: workers}
        manager = self.mturk_manager
        pool = manager._get_unique_pool(all_are_eligible)
        self.assertEqual(pool, [])
        manager._add_agent_to_pool(self.agent_1)
        manager._add_agent_to_pool(self.agent_2)
        manager._add_agent_to_pool(self.agent_3)
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1, self.agent_2, self.agent_3])
        manager._add_agent_to_pool(self.agent_1)
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1, self.agent_2, self.agent_3])
        manager._remove_from_agent_pool(self.agent_2)
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1, self.agent_3])
        with self.assertRaises(AssertionError):
            manager._remove_from_agent_pool(self.agent_2)
        second_worker_only = {'multiple': True, 'func': lambda workers: [workers[1]]}
        self.assertListEqual(manager._get_unique_pool(second_worker_only), [self.agent_3])
        only_agent_1 = {'multiple': False, 'func': lambda worker: worker is self.agent_1}
        self.assertListEqual(manager._get_unique_pool(only_agent_1), [self.agent_1])
        manager.force_expire_hit = mock.MagicMock()
        manager._expire_agent_pool()
        manager.force_expire_hit.assert_any_call(self.agent_1.worker_id, self.agent_1.assignment_id)
        manager.force_expire_hit.assert_any_call(self.agent_3.worker_id, self.agent_3.assignment_id)
        pool = manager._get_unique_pool(all_are_eligible)
        self.assertEqual(pool, [])
        self.agent_2.worker_id = self.agent_1.worker_id
        manager._add_agent_to_pool(self.agent_1)
        manager._add_agent_to_pool(self.agent_2)
        self.assertListEqual(manager.agent_pool, [self.agent_1, self.agent_2])
        manager.is_sandbox = False
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [self.agent_1])
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
def test_world_types(self):
    onboard_type = 'o_12345'
    waiting_type = 'w_12345'
    task_type = 't_12345'
    garbage_type = 'g_12345'
    manager = self.mturk_manager
    self.assertTrue(manager.is_onboarding_world(onboard_type))
    self.assertTrue(manager.is_task_world(task_type))
    self.assertTrue(manager.is_waiting_world(waiting_type))
    for world_type in [waiting_type, task_type, garbage_type]:
        self.assertFalse(manager.is_onboarding_world(world_type))
    for world_type in [onboard_type, task_type, garbage_type]:
        self.assertFalse(manager.is_waiting_world(world_type))
    for world_type in [waiting_type, onboard_type, garbage_type]:
        self.assertFalse(manager.is_task_world(world_type))
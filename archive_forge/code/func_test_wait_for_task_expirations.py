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
def test_wait_for_task_expirations(self):
    """
        Ensure waiting for expiration time actually works out.
        """
    manager = self.mturk_manager
    manager.opt['assignment_duration_in_seconds'] = 0.5
    manager.expire_all_unassigned_hits = mock.MagicMock()
    manager.update_hit_status = mock.MagicMock()
    manager.hit_id_list = [1, 2, 3]

    def run_task_wait():
        manager._wait_for_task_expirations()
    wait_thread = threading.Thread(target=run_task_wait, daemon=True)
    wait_thread.start()
    time.sleep(0.1)
    self.assertTrue(wait_thread.isAlive())
    assert_equal_by(wait_thread.isAlive, False, 3)
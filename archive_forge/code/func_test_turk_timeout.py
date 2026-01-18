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
def test_turk_timeout(self):
    """
        Timeout should send expiration message to worker and be treated as a disconnect
        event.
        """
    manager = self.mturk_manager
    manager.force_expire_hit = mock.MagicMock()
    manager._handle_agent_disconnect = mock.MagicMock()
    manager.handle_turker_timeout(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
    manager.force_expire_hit.assert_called_once()
    call_args = manager.force_expire_hit.call_args
    self.assertEqual(call_args[0][0], TEST_WORKER_ID_1)
    self.assertEqual(call_args[0][1], TEST_ASSIGNMENT_ID_1)
    manager._handle_agent_disconnect.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
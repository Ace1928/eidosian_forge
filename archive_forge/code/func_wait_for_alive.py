import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def wait_for_alive(self):
    last_time = time.time()
    while not self.ready:
        self.send_alive()
        time.sleep(0.5)
        assert time.time() - last_time < 10, 'Timed out wating for server to acknowledge {} alive'.format(self.worker_id)
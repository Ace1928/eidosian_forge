import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def assertEqualBy(self, val_func, val, max_time):
    start_time = time.time()
    while val_func() != val:
        assert time.time() - start_time < max_time, 'Value was not attained in specified time'
        time.sleep(0.1)
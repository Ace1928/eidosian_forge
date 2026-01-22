import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
class _FakeServer(object):
    """A fake server that runs a master session."""

    def start(self):
        logging.info('Creating a remote session to start a TensorFlow server, target = %r, session_config=%r', target, session_config)
        session.Session(target=target, config=session_config)

    def join(self):
        while True:
            time.sleep(5)
import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.dev.shared_utils import print_and_log
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def _send_world_alive(self):
    """
        Registers world with the passthrough server.
        """
    self._safe_send(json.dumps({'type': data_model.AGENT_ALIVE, 'content': {'id': 'WORLD_ALIVE', 'sender_id': self.get_my_sender_id()}}), force=True)
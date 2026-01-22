import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.shared_utils import print_and_log
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def _safe_put(self, connection_id, item):
    """
        Ensures that a queue exists before putting an item into it, logs if there's a
        failure.
        """
    if connection_id in self.queues:
        self.queues[connection_id].put(item)
    else:
        item[1].status = Packet.STATUS_FAIL
        shared_utils.print_and_log(logging.WARN, 'Queue {} did not exist to put a message in'.format(connection_id))
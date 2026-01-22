import copy
import queue
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import base
def _get_server_queue(self, topic, server):
    return self._server_queues.setdefault((topic, server), [])
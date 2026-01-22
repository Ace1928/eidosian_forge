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
def _get_num_workers(cluster_spec):
    """Gets number of workers including chief."""
    if not cluster_spec:
        return 0
    return len(cluster_spec.as_dict().get(_TaskType.WORKER, [])) + len(cluster_spec.as_dict().get(_TaskType.CHIEF, []))
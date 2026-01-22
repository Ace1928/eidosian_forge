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
def _is_chief(self):
    """Return whether the task is the chief worker."""
    if not self._cluster_spec or self._task_type in [_TaskType.CHIEF, _TaskType.EVALUATOR, None]:
        return True
    if _TaskType.CHIEF not in self._cluster_spec.jobs and self._task_type == _TaskType.WORKER and (self._task_id == 0):
        return True
    return False
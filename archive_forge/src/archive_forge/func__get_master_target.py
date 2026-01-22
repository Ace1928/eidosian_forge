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
def _get_master_target(self):
    """Return the master target for a task."""
    if not self._cluster_spec or self._task_type == _TaskType.EVALUATOR:
        return ''
    if not self._task_type:
        if _TaskType.CHIEF in self._cluster_spec.jobs:
            task_type = _TaskType.CHIEF
            task_id = 0
        else:
            assert _TaskType.WORKER in self._cluster_spec.jobs
            task_type = _TaskType.WORKER
            task_id = 0
    else:
        task_type = self._task_type
        task_id = self._task_id
    prefix = ''
    if self._rpc_layer:
        prefix = self._rpc_layer + '://'
    return prefix + self._cluster_spec.job_tasks(task_type)[task_id or 0]
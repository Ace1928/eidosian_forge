import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def update_worker_num_paused_threads(self, worker_id, num_paused_threads_delta):
    """Updates the number of paused threads of a worker.

        Args:
            worker_id: ID of this worker. Type is bytes.
            num_paused_threads_delta: The delta of the number of paused threads.

        Returns:
             Is operation success
        """
    self._check_connected()
    assert worker_id is not None, 'worker_id is not valid'
    assert num_paused_threads_delta is not None, 'worker_id is not valid'
    return self.global_state_accessor.update_worker_num_paused_threads(worker_id, num_paused_threads_delta)
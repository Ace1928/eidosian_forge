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
def workers(self):
    """Get a dictionary mapping worker ID to worker information."""
    self._check_connected()
    worker_table = self.global_state_accessor.get_worker_table()
    workers_data = {}
    for i in range(len(worker_table)):
        worker_table_data = gcs_pb2.WorkerTableData.FromString(worker_table[i])
        if worker_table_data.is_alive and worker_table_data.worker_type == common_pb2.WORKER:
            worker_id = binary_to_hex(worker_table_data.worker_address.worker_id)
            worker_info = worker_table_data.worker_info
            workers_data[worker_id] = {'node_ip_address': decode(worker_info[b'node_ip_address']), 'plasma_store_socket': decode(worker_info[b'plasma_store_socket'])}
            if b'stderr_file' in worker_info:
                workers_data[worker_id]['stderr_file'] = decode(worker_info[b'stderr_file'])
            if b'stdout_file' in worker_info:
                workers_data[worker_id]['stdout_file'] = decode(worker_info[b'stdout_file'])
    return workers_data
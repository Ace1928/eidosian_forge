import functools
import os
import threading
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as base_cluster_resolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import server_lib
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def _set_num_gpus(self):
    devices = config.list_logical_devices('GPU')
    per_worker_gpus = {}
    for d in devices:
        d_spec = tf_device.DeviceSpec.from_string(d.name)
        if d_spec.device_type == 'GPU' and d_spec.job == 'worker':
            job_spec = d_spec.replace(device_type=None, device_index=None)
            per_worker_gpus[job_spec] = per_worker_gpus.get(job_spec, 0) + 1
    num_gpus = 0
    for _, count in per_worker_gpus.items():
        if num_gpus > 0 and count != num_gpus:
            raise ValueError('Mismatched number of GPUs per worker')
        num_gpus = count
    self._num_gpus_per_worker = num_gpus
    logging.info(f'Number of GPUs on workers: {self._num_gpus_per_worker}')
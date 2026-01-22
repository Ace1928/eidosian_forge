import copy
import threading
import time
import weakref
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def _initialize_local(self, cluster_resolver, devices=None):
    """Initializes the object for local training."""
    self._is_chief = True
    self._num_workers = 1
    if ops.executing_eagerly_outside_functions():
        try:
            context.context().configure_collective_ops(scoped_allocator_enabled_ops=('CollectiveReduce',))
        except RuntimeError:
            logging.warning('Collective ops is not configured at program startup. Some performance features may not be enabled.')
        self._collective_ops_configured = True
    if devices:
        local_devices = devices
        if 'GPU' in devices[0]:
            local_device_type = 'GPU'
        elif 'TPU' in devices[0]:
            local_device_type = 'TPU'
        else:
            local_device_type = 'CPU'
    else:
        local_devices, local_device_type = self._initialize_local_devices(cluster_resolver, worker_device='')
    self._worker_device = device_util.canonicalize('/device:CPU:0')
    self._host_input_device = numpy_dataset.SingleDevice(self._worker_device)
    self._collective_keys = cross_device_utils.CollectiveKeys(group_key_start=1 + self._collective_key_base)
    self._cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(devices=local_devices, group_size=len(local_devices), options=self._communication_options, collective_keys=self._collective_keys)
    self._host_cross_device_ops = cross_device_ops_lib.CollectiveAllReduce(devices=[self._worker_device], group_size=self._num_workers, options=self._communication_options, collective_keys=self._collective_keys)
    super(CollectiveAllReduceExtended, self)._initialize_single_worker(local_devices)
    self._cluster_spec = None
    self._task_type = None
    self._task_id = None
    self._id_in_cluster = 0
    self._local_or_standalone_client_mode = True
    self._num_devices_per_worker = len(local_devices)
    self._local_device_type = local_device_type
    self._rpc_layer = cluster_resolver.rpc_layer
    self._warn_nccl_no_gpu()
    logging.info('Single-worker MultiWorkerMirroredStrategy with local_devices = %r, communication = %s', local_devices, self._communication_options.implementation)
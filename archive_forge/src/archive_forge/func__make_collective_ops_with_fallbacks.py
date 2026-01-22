import copy
from tensorflow.python import tf2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _make_collective_ops_with_fallbacks(self):
    self._collective_keys = cross_device_utils.CollectiveKeys(group_key_start=1 + self._collective_key_base)
    if not ops.executing_eagerly_outside_functions() and any(('gpu' not in d.lower() for d in self._devices)):
        return cross_device_ops_lib.ReductionToOneDevice()
    if any(('cpu' in d.lower() for d in self._devices)) and any(('gpu' in d.lower() for d in self._devices)):
        return cross_device_ops_lib.ReductionToOneDevice()
    if all(('cpu' in d.lower() for d in self._devices)):
        self._communication_options = collective_util.Options(implementation=collective_util.CommunicationImplementation.RING)
    else:
        physical_gpus = context.context().list_physical_devices(device_type='GPU')
        logical_gpus = context.context().list_logical_devices(device_type='GPU')
        if len(physical_gpus) < len(logical_gpus):
            self._communication_options = collective_util.Options(implementation=collective_util.CommunicationImplementation.RING)
    return cross_device_ops_lib.CollectiveAllReduce(devices=self._devices, group_size=len(self._devices), options=self._communication_options, collective_keys=self._collective_keys)
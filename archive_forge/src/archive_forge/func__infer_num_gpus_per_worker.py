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
def _infer_num_gpus_per_worker(devices):
    """Infers the number of GPUs on each worker.

  Currently to make multi-worker cross device ops work, we need all workers to
  have the same number of GPUs.

  Args:
    devices: a list of device strings, can be either local devices or remote
      devices.

  Returns:
    number of GPUs per worker.

  Raises:
    ValueError if workers have different number of GPUs or GPU indices are not
    consecutive and starting from 0.
  """
    if _is_device_list_single_worker(devices):
        return sum((1 for d in devices if _is_gpu_device(d)))
    else:
        device_dict = _group_device_list(devices)
        num_gpus = None
        for _, devices_in_task in device_dict.items():
            for device_in_task in devices_in_task:
                if num_gpus is None:
                    num_gpus = sum((1 for d in device_in_task if _is_gpu_device(d)))
                elif num_gpus != sum((1 for d in device_in_task if _is_gpu_device(d))):
                    raise ValueError('All workers should have the same number of GPUs.')
                for d in device_in_task:
                    d_spec = tf_device.DeviceSpec.from_string(d)
                    if d_spec.device_type == 'GPU' and d_spec.device_index >= num_gpus:
                        raise ValueError('GPU `device_index` on a worker should be consecutive and start from 0.')
        return num_gpus
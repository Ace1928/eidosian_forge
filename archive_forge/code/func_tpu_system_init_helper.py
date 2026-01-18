import functools
import time
from typing import List, Optional, Dict
import numpy as np
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.util.tf_export import tf_export
def tpu_system_init_helper(task_id, num_tasks, num_devices, use_tfrt_host_runtime=True):
    """A helper function to initialize multi-client tpu system."""

    @def_function.function
    def _tpu_init_fn():
        return gen_dtensor_ops.configure_and_initialize_global_tpu(use_tfrt_host_runtime=use_tfrt_host_runtime)

    @def_function.function
    def _set_global_tpu_array_fn(topology_proto):
        gen_dtensor_ops.d_tensor_set_global_tpu_array(topology_proto)
    with ops.device('/job:' + config.full_job_name() + '/device:TPU_SYSTEM:0'):
        my_core_ids = _tpu_init_fn()
    logging.info('TPU core IDs: %s', my_core_ids)
    num_devices_per_task = int(num_devices / num_tasks)
    mesh = layout_lib.Mesh([_MESH_DIM_X], *_create_device_array((num_devices,), _TPU_DEVICE_TYPE, config.client_id()))
    layout = layout_lib.Layout([_MESH_DIM_X, layout_lib.UNSHARDED], mesh)
    device = dtensor_device.DTensorDevice(meshes=[mesh])
    logging.info('TPU core locations: %s', device.tpu_core_ids_to_locations(my_core_ids))
    all_core_ids = np.zeros([num_devices], dtype=np.int32)
    for i in range(len(my_core_ids)):
        all_core_ids[task_id * num_devices_per_task + i] = my_core_ids[i]
    all_core_ids = constant_op.constant([all_core_ids])
    zeros = array_ops.zeros_like(all_core_ids)
    all_core_ids = [all_core_ids] + [zeros] * (num_devices_per_task - 1)
    with ops.device(device.name):
        all_core_ids = device.pack(all_core_ids, layout)
        all_core_ids = math_ops.reduce_sum(all_core_ids, axis=[0])
        unpacked_all_tpu_ids = device.unpack(all_core_ids)
    all_core_ids = list(unpacked_all_tpu_ids[0].numpy())
    logging.info('All TPU core IDs: %s', all_core_ids)
    device.set_tpu_core_ids('', all_core_ids)
    global _all_core_ids
    _all_core_ids = all_core_ids
    all_core_locations = device.tpu_core_ids_to_locations(all_core_ids)
    all_core_locations = [_CoreLocation(l[0], l[1], l[2], l[3]) for l in all_core_locations]
    global _all_core_locations
    _all_core_locations = all_core_locations
    logging.info('All TPU core locations: %s', all_core_locations)
    tpu_topology = _create_tpu_topology(all_core_locations, num_tasks, num_devices_per_task)
    _set_global_tpu_array_fn(tpu_topology.serialized())
    return (tpu_topology, device)
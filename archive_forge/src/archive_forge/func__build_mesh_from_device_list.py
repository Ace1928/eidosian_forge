from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import device as tf_device
@classmethod
def _build_mesh_from_device_list(cls, devices):
    if devices:
        device_type = tf_device.DeviceSpec.from_string(devices[0]).device_type
        dtensor_util.initialize_accelerator_system_once(device_type)
        mesh = mesh_util.create_mesh(mesh_dims=[(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, len(devices))], devices=devices)
    else:
        device_type = d_config.preferred_device_type()
        devices = d_config.local_devices(device_type)
        dtensor_util.initialize_accelerator_system_once(device_type)
        mesh = mesh_util.create_mesh(mesh_dims=[(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, len(devices))], device_type=device_type)
    return mesh
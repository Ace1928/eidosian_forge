from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import device as tf_device
@classmethod
def _validate_init_args(cls, mesh, devices):
    if mesh and devices:
        raise ValueError(f'Mesh and devices can not be provided at the same time. received mesh = {mesh}, devices = {devices}')
    if mesh and len(mesh.shape()) != 1:
        raise ValueError(f'The mesh for MirroredStrategy must be 1D, received: {len(mesh.shape())}D')
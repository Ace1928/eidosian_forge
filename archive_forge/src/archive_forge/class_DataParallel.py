import collections
import contextlib
import os
import re
import warnings
import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import distribution_lib
from keras.src.backend.common import global_state
@keras_export('keras.distribution.DataParallel')
class DataParallel(Distribution):
    """Distribution for data parallelism.

    You can choose to create this instance by either specifing
    the `device_mesh` or `devices` arguments (but not both).

    The `device_mesh` argument is expected to be a `DeviceMesh` instance,
    and is expected to be 1D only. In case that the mesh has multiple axes,
    then the first axis will be treated as the data parallel dimension
    (and a warning will be raised).

    When a list of `devices` are provided, they will be used to construct a
    1D mesh.

    When both `mesh` and `devices` are absent, then `list_devices()`
    will be used to detect any available devices and create a 1D mesh from
    them.

    Args:
        device_mesh: Optional `DeviceMesh` instance.
        devices: Optional list of devices.
    """

    def __init__(self, device_mesh=None, devices=None):
        if device_mesh:
            self._initialize_with_device_mesh(device_mesh)
        elif devices:
            self._initialize_mesh_from_devices(devices)
        else:
            self._initialize_mesh_from_list_devices()
        self._batch_dim_name = self.device_mesh.axis_names[0]
        self._num_process = distribution_lib.num_processes()
        self._process_id = distribution_lib.process_id()
        self._is_multi_process = self._num_process > 1

    def _initialize_with_device_mesh(self, device_mesh):
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(f'Expect `mesh` to be an instance of `DeviceMesh`. Received: mesh={device_mesh} (of type {type(device_mesh)})')
        super().__init__(device_mesh)
        if self.device_mesh.devices.ndim != 1:
            warnings.warn('Expect the input mesh to be 1D, but received mesh.devices.ndim=%d. The first axis will be used for data-parallel sharding.', device_mesh.devices.ndim)

    def _initialize_mesh_from_devices(self, devices):
        devices = np.array(devices)
        device_mesh = DeviceMesh(shape=devices.shape, axis_names=[DEFAULT_BATCH_DIM_NAME], devices=devices)
        super().__init__(device_mesh)

    def _initialize_mesh_from_list_devices(self):
        devices = np.array(list_devices())
        device_mesh = DeviceMesh(shape=devices.shape, axis_names=[DEFAULT_BATCH_DIM_NAME], devices=devices)
        super().__init__(device_mesh)

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self._batch_dim_name
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

    def get_tensor_layout(self, path):
        return None

    def distribute_dataset(self, dataset):
        from tensorflow.python.data.experimental.ops import distribute as tf_data_distribute
        from keras.src.utils.module_utils import tensorflow as tf
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError(f'Only `tf.data.Dataset` is supported for sharding, got {type(dataset)}')
        if not self._is_multi_process:
            return dataset
        batch_size = tf_data_distribute.compute_batch_size(dataset)
        if batch_size.numpy() < 0:
            raise ValueError('The batch size of the input dataset is unknown. Please config the batch size for the input dataset, e.g via `dataset.batch(batch_size)`')
        per_worker_batch_size = tf_data_distribute.batch_sizes_for_worker(global_batch_size=batch_size, num_workers=self._num_process, num_replicas_per_worker=1, worker_index=self._process_id)
        distributed_dataset = dataset.rebatch(per_worker_batch_size)
        distributed_dataset = tf_data_distribute._AutoShardDataset(distributed_dataset, num_workers=self._num_process, index=self._process_id, num_replicas=self._num_process)
        return distributed_dataset.prefetch(tf.data.AUTOTUNE)
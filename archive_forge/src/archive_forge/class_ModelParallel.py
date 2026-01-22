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
@keras_export('keras.distribution.ModelParallel')
class ModelParallel(Distribution):
    """Distribution that shards model variables.

    Compare to `DataParallel` which replicates the variables across all devices,
    `ModelParallel` allows you to shard variables in addition to the input data.

    To construct a `ModelParallel` distribution, you need to provide a
    `DeviceMesh` and a `LayoutMap`.

    1. `DeviceMesh` contains physcial device information. The axis names in
        the mesh will be used to map the variable and data layout.
    2. `LayoutMap` contains the mapping between variable paths to their
        corresponding `TensorLayout`.

    Example:

    ```python
    devices = list_devices()    # Assume there are 8 devices.

    # Create a mesh with 2 devices for data parallelism and 4 devices for
    # model parallelism.
    device_mesh = DeviceMesh(shape=(2, 4), axis_names=('batch', 'model'),
                             devices=devices)
    # Create a layout map that shard the `Dense` layer and `Conv2D`
    # layer variables on the last dimension.
    # Based on the `device_mesh`, this means the variables
    # will be split across 4 devices. Any other variable that doesn't
    # match any key in the layout map will be fully replicated.
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    layout_map['conv2d.*kernel'] = (None, None, None, 'model')
    layout_map['conv2d.*bias'] = ('model',)

    distribution = ModelParallel(device_mesh=device_mesh,
                                 layout_map=layout_map,
                                 batch_dim_name='batch')
    # Set the global distribution, or via `with distribution.scope():`
    set_distribution(distribution)

    model = model_creation()
    model.compile()
    model.fit(data)
    ```

    You can quickly update the device mesh shape to change the sharding factor
    of the variables. E.g.
    ```
    # With only the shape change for the device mesh, the variables will be
    # sharded across 8 devices instead of 4, which further reduces the memory
    # footprint of variables on each of the device.
    device_mesh = DeviceMesh(shape=(1, 8), axis_names=('batch', 'model'),
                             devices=devices)
    ```

    To figure out a proper layout mapping rule for all the model variables, you
    can first list out all the model variable paths, which will be used as the
    key to map the variables to `TensorLayout`.

    e.g.
    ```
    model = create_model()
    for v in model.variables:
        print(v.path)
    ```

    Args:
        device_mesh: `DeviceMesh` instance for physical device and its
            logical mapping.
        layout_map: `LayoutMap` instance which map the variable path to the
            corresponding `TensorLayout`. The axis names of the
            `TensorLayout`s should match to the axis names in the
            device_mesh, or exception will be raised.
        batch_dim_name: optional string, the axis name in the `device_mesh`
            that will be used to distribute data. If unspecified, the
            first axis from the `device_mesh` will be used.
    """

    def __init__(self, device_mesh, layout_map, batch_dim_name=None):
        super().__init__(device_mesh)
        self._layout_map = layout_map
        self._batch_dim_name = batch_dim_name or self.device_mesh.axis_names[0]
        self._num_process = distribution_lib.num_processes()
        self._process_id = distribution_lib.process_id()
        self._is_multi_process = self._num_process > 1

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self._batch_dim_name
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        variable_layout = self._layout_map[variable.path]
        if variable_layout is not None:
            return variable_layout
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

    def get_tensor_layout(self, path):
        return self._layout_map[path]

    def distribute_dataset(self, dataset):
        from tensorflow.python.data.experimental.ops import distribute as tf_data_distribute
        from keras.src.utils.module_utils import tensorflow as tf
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError(f'Only `tf.data.Dataset` is supported for sharding, got {type(dataset)}')
        if not self._is_multi_process:
            return dataset
        global_batch_size = tf_data_distribute.compute_batch_size(dataset)
        if global_batch_size.numpy() < 0:
            raise ValueError('The batch size of the input dataset is unknown. Please config the batch size for the input dataset, e.g via `dataset.batch(batch_size)`')
        mesh_batch_dim_index = self.device_mesh.axis_names.index(self._batch_dim_name)
        mesh_batch_dim_size = self.device_mesh.shape[mesh_batch_dim_index]
        local_device_count = np.prod(self.device_mesh.shape) // self._num_process
        if mesh_batch_dim_size < local_device_count:
            return dataset.prefetch(tf.data.AUTOTUNE)
        else:
            if mesh_batch_dim_size % local_device_count != 0:
                raise ValueError(f'The Batch dimention of the mesh is not compatible with the local worker device count. Mesh batch dim = {mesh_batch_dim_size} and local device count = {local_device_count}')
            num_shards = mesh_batch_dim_size // local_device_count
            per_worker_batch_size = global_batch_size // num_shards
            distributed_dataset = dataset.rebatch(per_worker_batch_size)
            distributed_dataset = tf_data_distribute._AutoShardDataset(distributed_dataset, num_workers=num_shards, index=self._process_id % num_shards, num_replicas=num_shards)
            return distributed_dataset.prefetch(tf.data.AUTOTUNE)
from typing import Dict, List, Optional
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class _DSaver:
    """A single device saver that places tensors on DTensor Device."""

    def __init__(self, mesh: layout.Mesh, saveable_objects: List[saveable_object.SaveableObject]):
        self._saveable_objects = saveable_objects
        self._mesh = mesh

    def save(self, file_prefix: str, options: Optional[checkpoint_options.CheckpointOptions]=None) -> Optional[ops.Operation]:
        """Saves the saveable objects to a checkpoint with `file_prefix`.

    Also query the generated shards from the distributed DTensor SaveV2 ops and
    do a MergeV2 on those. Each op here is backed by a global_barrier to avoid
    racing from multiple clients.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix to
        save under.
      options: Optional `CheckpointOptions` object. This is unused in DTensor.

    Returns:
      An `Operation`, or None when executing eagerly.
    """
        if options is not None and options.experimental_io_device is not None:
            raise ValueError('Specified experimental_io_device in DTensor checkpoint is not supported.')
        del options
        tensor_names = []
        tensors = []
        tensor_slices = []
        for saveable in self._saveable_objects:
            for spec in saveable.specs:
                tensor = spec.tensor
                if tensor is not None:
                    if api.device_name() != spec.device:
                        tensor = api.pack([tensor] * self._mesh.host_mesh().num_local_devices(), layout.Layout.replicated(self._mesh.host_mesh(), rank=tensor.shape.rank))
                    tensor_names.append(spec.name)
                    tensors.append(tensor)
                    tensor_slices.append(spec.slice_spec)
        return save_restore.sharded_save(self._mesh, file_prefix, tensor_names, tensor_slices, tensors)

    def restore(self, file_prefix: str, options: Optional[checkpoint_options.CheckpointOptions]=None) -> Dict[str, ops.Operation]:
        """Restore the saveable objects from a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix for
        files to read from.
      options: Optional `CheckpointOptions` object. This is unused in DTensor.

    Returns:
      A dictionary mapping from SaveableObject names to restore operations.
    """
        if options is not None and options.experimental_io_device is not None:
            raise ValueError('Specified experimental_io_device in DTensor checkpoint is not supported.')
        del options
        restore_specs = []
        tensor_structure = []
        for saveable in self._saveable_objects:
            saveable_tensor_structure = []
            tensor_structure.append(saveable_tensor_structure)
            for spec in saveable.specs:
                saveable_tensor_structure.append(spec.name)
                if isinstance(spec, d_variable.DSaveSpec):
                    restore_specs.append((spec.name, spec.slice_spec, spec.dtype, spec.layout, spec.global_shape))
                elif isinstance(spec, saveable_object.SaveSpec):
                    restore_specs.append((spec.name, spec.slice_spec, spec.dtype, layout.Layout.replicated(self._mesh.host_mesh(), spec.tensor.shape.rank).to_string(), spec.tensor.shape.as_list()))
        tensor_names, tensor_slices, tensor_dtypes, layouts, global_shapes = zip(*restore_specs)
        with ops.device(api.device_name()):
            restored_tensors = gen_dtensor_ops.d_tensor_restore_v2(prefix=file_prefix, tensor_names=tensor_names, shape_and_slices=tensor_slices, input_shapes=global_shapes, input_layouts=layouts, dtypes=tensor_dtypes)
        structured_restored_tensors = nest.pack_sequence_as(tensor_structure, restored_tensors)
        restore_ops = {}
        for saveable, restored_tensors in zip(self._saveable_objects, structured_restored_tensors):
            restore_ops[saveable.name] = saveable.restore(restored_tensors, restored_shapes=None)
        return restore_ops
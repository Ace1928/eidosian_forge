from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
class MultiDeviceSaver(object):
    """Saves checkpoints directly from multiple devices.

  Note that this is a low-level utility which stores Tensors in the keys
  specified by `SaveableObject`s. Higher-level utilities for object-based
  checkpointing are built on top of it.
  """

    def __init__(self, serialized_tensors, registered_savers=None, call_with_mapped_captures=None):
        """Specify a list of `SaveableObject`s to save and restore.

    Args:
      serialized_tensors: A dictionary mapping `Trackable` to a tensor dict,
        which maps checkpoint_key -> (slice_spec ->) -> Tensor/SaveSpec. The
        `Trackable` key is used to get the `restore_from_tensors` function,
        and may be `None` if the tensor is not meant to be restored.
      registered_savers: A dictionary mapping `registration.RegisteredSaver`
        namedtuples to a dictionary of named Trackables. The keys of the
        Trackable dictionary are string names that uniquely identify the
        Trackable in the checkpoint.
      call_with_mapped_captures: TODO
    """
        self._keys_to_restore_fn = {}
        self._restore_fn_to_keys = {}
        tensors_by_device = {}
        for obj, tensor_dict in serialized_tensors.items():
            restore_fn = _restore_noop if obj is None else obj._restore_from_tensors
            for checkpoint_key, maybe_tensor in tensor_dict.items():
                if not isinstance(maybe_tensor, dict):
                    maybe_tensor = {'': maybe_tensor}
                for slice_spec, tensor in maybe_tensor.items():
                    if (checkpoint_key, slice_spec) in self._keys_to_restore_fn:
                        raise ValueError('Recieved multiple tensors with the same checkpoint key and slice spec. This is invalid because one will overwrite the other in the checkpoint. This indicates a bug in the Checkpoint key-generation.')
                    self._keys_to_restore_fn[checkpoint_key, slice_spec] = restore_fn
                    self._restore_fn_to_keys.setdefault(restore_fn, []).append((checkpoint_key, slice_spec))
                    host_device = saveable_object_util.set_cpu0(tensor.device)
                    tensors_by_device.setdefault(host_device, {}).setdefault(checkpoint_key, {})[slice_spec] = tensor
        self._single_device_savers = {device: _SingleDeviceSaver(tensor_slice_dict) for device, tensor_slice_dict in tensors_by_device.items()}
        self._registered_savers = {}
        if registered_savers:
            for registered_name, trackables in registered_savers.items():
                save_fn = _get_mapped_registered_save_fn(registration.get_save_function(registered_name), trackables, call_with_mapped_captures)
                restore_fn = _get_mapped_registered_restore_fn(registration.get_restore_function(registered_name), trackables, call_with_mapped_captures)
                self._registered_savers[registered_name] = (save_fn, restore_fn)

    @classmethod
    def from_saveables(cls, saveables, registered_savers=None, call_with_mapped_captures=None):
        serialized_tensors = object_identity.ObjectIdentityDictionary()
        for saveable in saveables:
            trackable = saveable_object_util.SaveableCompatibilityConverter(saveable, saveables=[saveable])
            serialized_tensors[trackable] = trackable._serialize_to_tensors()
        return cls(serialized_tensors, registered_savers, call_with_mapped_captures)

    def to_proto(self):
        """Serializes to a SaverDef referencing the current graph."""
        filename_tensor = array_ops.placeholder(shape=[], dtype=dtypes.string, name='saver_filename')
        save_tensor = self._traced_save(filename_tensor)
        restore_op = self._traced_restore(filename_tensor).op
        return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.name, save_tensor_name=save_tensor.name, restore_op_name=restore_op.name, version=saver_pb2.SaverDef.V2)

    @def_function.function(input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),), autograph=False)
    def _traced_save(self, file_prefix):
        save_op = self.save(file_prefix)
        with ops.device('cpu:0'):
            with ops.control_dependencies([save_op]):
                return array_ops.identity(file_prefix)

    @def_function.function(input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),), autograph=False)
    def _traced_restore(self, file_prefix):
        restore_ops = self.restore(file_prefix)
        with ops.device('cpu:0'):
            with ops.control_dependencies(restore_ops.values()):
                return array_ops.identity(file_prefix)

    def save(self, file_prefix, options=None):
        """Save the saveable objects to a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix to
        save under.
      options: Optional `CheckpointOptions` object.
    Returns:
      An `Operation`, or None when executing eagerly.
    """
        options = options or checkpoint_options.CheckpointOptions()
        with ops.device('CPU'):
            sharded_suffix = array_ops.where(string_ops.regex_full_match(file_prefix, '^s3://.*'), constant_op.constant('.part'), constant_op.constant('_temp/part'))
            tmp_checkpoint_prefix = string_ops.string_join([file_prefix, sharded_suffix])
            registered_paths = {saver_name: registered_saver_filename(file_prefix, saver_name) for saver_name in self._registered_savers}

        def save_fn():
            saved_prefixes = []
            for saver_name, (save_fn, _) in self._registered_savers.items():
                maybe_saved_prefixes = save_fn(registered_paths[saver_name])
                if maybe_saved_prefixes is not None:
                    flattened_saved_prefixes = nest.flatten(maybe_saved_prefixes)
                    if not all((tensor_util.is_tf_type(x) and x.dtype == dtypes.string for x in flattened_saved_prefixes)):
                        raise ValueError(f'Registered saver must return a (maybe empty) list of string type tensors. Got {maybe_saved_prefixes}.')
                    saved_prefixes.extend(flattened_saved_prefixes)
            num_shards = len(self._single_device_savers)
            sharded_saves = []
            num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
            last_device = None
            for shard, (device, saver) in enumerate(sorted(self._single_device_savers.items())):
                last_device = device
                with ops.device(saveable_object_util.set_cpu0(device)):
                    shard_prefix = sharded_filename(tmp_checkpoint_prefix, shard, num_shards_tensor)
                saved_prefixes.append(shard_prefix)
                with ops.device(device):
                    sharded_saves.append(saver.save(shard_prefix, options))
            with ops.control_dependencies(sharded_saves):
                merge_device = options.experimental_io_device or saveable_object_util.set_cpu0(last_device)
                with ops.device(merge_device):
                    return gen_io_ops.merge_v2_checkpoints(saved_prefixes, file_prefix, delete_old_dirs=True)
        if context.executing_eagerly() and len(self._single_device_savers) > 1:

            @def_function.function(jit_compile=False)
            def tf_function_save():
                save_fn()
            tf_function_save()
        else:
            return save_fn()

    def restore(self, file_prefix, options=None):
        """Restore the saveable objects from a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix for
        files to read from.
      options: Optional `CheckpointOptions` object.

    Returns:
      When not run eagerly or when saving on a single device, returns a
      dictionary mapping from SaveableObject names to restore operations;
      otherwise, returns an empty dict.
    """
        options = options or checkpoint_options.CheckpointOptions()

        def restore_fn():
            restore_fn_inputs = {}
            restore_fn_input_count = {fn: len(keys) for fn, keys in self._restore_fn_to_keys.items()}
            restore_ops = {}
            for device, saver in sorted(self._single_device_savers.items()):
                with ops.device(device):
                    restored_tensor_dict = saver.restore(file_prefix, options)
                    for checkpoint_key, slice_and_tensor in restored_tensor_dict.items():
                        for slice_spec, tensor in slice_and_tensor.items():
                            restore_fn = self._keys_to_restore_fn[checkpoint_key, slice_spec]
                            if slice_spec:
                                restore_fn_inputs.setdefault(restore_fn, {}).setdefault(checkpoint_key, {})[slice_spec] = tensor
                            else:
                                restore_fn_inputs.setdefault(restore_fn, {})[checkpoint_key] = tensor
                            restore_fn_input_count[restore_fn] -= 1
                            if restore_fn_input_count[restore_fn] == 0:
                                restored_tensors = {}
                                for ckpt_key, tensor in restore_fn_inputs[restore_fn].items():
                                    restored_tensors[trackable_utils.extract_local_name(ckpt_key)] = tensor
                                ret = restore_fn(restored_tensors)
                                if isinstance(ret, dict):
                                    restore_ops.update(ret)
            for _, (_, restore_fn) in self._registered_savers.items():
                restore_fn(file_prefix)
            return restore_ops
        has_custom_device_saver = any([context.is_custom_device(d) for d in self._single_device_savers.keys()])
        if context.executing_eagerly() and (len(self._single_device_savers) > 1 or has_custom_device_saver):

            @def_function.function(jit_compile=False, autograph=False)
            def tf_function_restore():
                restore_fn()
                return {}
            restore_ops = tf_function_restore()
        else:
            restore_ops = restore_fn()
        return restore_ops
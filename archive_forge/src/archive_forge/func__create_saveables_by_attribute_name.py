import collections
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import constants
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def _create_saveables_by_attribute_name(self, saveable_factories):
    """Creates or caches SaveableObjects by matching the attribute names.

    The attribute name keys in the `saveable_factories` is used to find the
    corresponding attribute in the object proto. Attributes contain checkpoint
    keys which are passed to the factory function to generate the
    SaveableObject.

    Args:
      saveable_factories: a dict mapping attribute name to a callable factory
        function that produces a SaveableObject.

    Returns:
      A tuple of (
          existing_restore_ops: list,
          named_saveables: dict)
    """
    named_saveables = {}
    existing_restore_ops = []
    created_compat_names = set()
    for serialized_tensor in self.object_proto.attributes:
        if context.executing_eagerly():
            existing_op = None
        else:
            existing_op = self._checkpoint.restore_ops_by_name.get(serialized_tensor.checkpoint_key, None)
        if existing_op is not None:
            existing_restore_ops.append(existing_op)
            continue
        if any((serialized_tensor.name.startswith(name) for name in created_compat_names)):
            continue
        saveables_cache = self._checkpoint.saveables_cache
        if saveables_cache is None:
            saveable = None
        else:
            saveable_list = saveables_cache.get(self.trackable, {}).get(serialized_tensor.name, (None,))
            if len(saveable_list) == 1:
                saveable, = saveable_list
            else:
                saveable = None
        if saveable is not None:
            if serialized_tensor.checkpoint_key not in saveable.name:
                saveable = None
                del saveables_cache[self.trackable]
        if saveable is None:
            saveable = _get_saveable_from_factory(saveable_factories, serialized_tensor, created_compat_names)
            if saveable is None:
                self._checkpoint.unused_attributes.setdefault(self._proto_id, []).append(serialized_tensor.name)
                continue
            if saveables_cache is not None:
                saveables_cache.setdefault(self.trackable, {})[serialized_tensor.name] = [saveable]
        named_saveables[serialized_tensor.checkpoint_key] = saveable
    return (existing_restore_ops, named_saveables)
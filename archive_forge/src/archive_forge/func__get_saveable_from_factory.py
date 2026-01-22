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
def _get_saveable_from_factory(saveable_factories, serialized_tensor, created_compat_names):
    """Returns the saveable generated from the factory method."""
    matched_factory = None
    expected_factory_name = serialized_tensor.name
    factory_input_name = serialized_tensor.checkpoint_key
    if expected_factory_name in saveable_factories:
        matched_factory = saveable_factories[expected_factory_name]
    if matched_factory is None:
        for factory_name, factory in saveable_factories.items():
            if expected_factory_name.startswith(factory_name):
                if matched_factory is not None:
                    raise ValueError('Forward compatibility load error: Unable to load checkpoint saved in future version of TensorFlow. Please update your version of TensorFlow to the version in which the checkpoint was saved.')
                matched_factory = factory
                factory_input_name = _extract_saveable_name(serialized_tensor.checkpoint_key) + factory_name
                created_compat_names.add(factory_name)
    if callable(matched_factory):
        return matched_factory(name=factory_input_name)
    return matched_factory
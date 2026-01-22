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
def _queue_slot_variables(checkpoint_position, visit_queue):
    """Queues slot variables for restoration."""
    trackable = checkpoint_position.trackable
    checkpoint = checkpoint_position.checkpoint
    for deferred_slot_restoration in checkpoint.deferred_slot_restorations.pop(checkpoint_position.proto_id, ()):
        slot_variable_position, slot_variable = checkpoint_position.create_slot_variable_position(trackable, deferred_slot_restoration.original_variable, deferred_slot_restoration.slot_variable_id, deferred_slot_restoration.slot_name)
        if slot_variable_position is not None:
            visit_queue.append((slot_variable_position, slot_variable))
    for slot_restoration in checkpoint.slot_restorations.pop(checkpoint_position.proto_id, ()):
        optimizer_object = checkpoint.object_by_proto_id.get(slot_restoration.optimizer_id, None)
        if optimizer_object is None:
            checkpoint.deferred_slot_restorations.setdefault(slot_restoration.optimizer_id, []).append(_DeferredSlotVariableRestoration(original_variable=trackable, slot_variable_id=slot_restoration.slot_variable_id, slot_name=slot_restoration.slot_name))
        elif hasattr(optimizer_object, '_create_or_restore_slot_variable'):
            slot_variable_position, slot_variable = checkpoint_position.create_slot_variable_position(optimizer_object, trackable, slot_restoration.slot_variable_id, slot_restoration.slot_name)
            if slot_variable_position is not None:
                visit_queue.append((slot_variable_position, slot_variable))
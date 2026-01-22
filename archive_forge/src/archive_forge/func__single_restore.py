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
def _single_restore(self):
    """Restores the trackable."""
    trackable = self.trackable
    trackable._maybe_initialize_trackable()
    checkpoint = self.checkpoint
    if checkpoint.restore_uid > trackable._update_uid:
        restore_ops, tensor_saveables, python_positions, registered_savers = self.gather_ops_or_named_saveables()
        trackable._update_uid = checkpoint.restore_uid
    else:
        restore_ops = ()
        tensor_saveables = {}
        python_positions = ()
        registered_savers = {}
    return (restore_ops, tensor_saveables, python_positions, registered_savers)
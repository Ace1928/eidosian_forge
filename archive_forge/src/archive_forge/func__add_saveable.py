import functools
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _add_saveable(saveables, seen_ops, saveable):
    """Adds the saveable to the saveables list.

  Args:
    saveables: List to append the SaveableObject to.
    seen_ops: Set of the ops of the saveables already processed.  Used to
      check that each saveable is only saved once.
    saveable: The saveable.

  Raises:
    ValueError: If the saveable has already been processed.
  """
    if saveable.op is not None and saveable.op in seen_ops:
        raise ValueError(f'The same saveable will be restored with two names: {saveable.name}')
    saveables.append(saveable)
    seen_ops.add(saveable.op)
import collections
import contextlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@contextlib.contextmanager
def _push_critical_section_stack(signature):
    """Push a CriticalSection._signature to the thread-local stack.

  If the signature is already on the stack, raise an error because it means
  we're trying to execute inside the same locked CriticalSection, which
  will create a deadlock.

  Args:
    signature: Tuple of the type `CriticalSection._signature`.  Uniquely
      identifies a CriticalSection by its `shared_name`, `container`,
      and device.

  Yields:
    An empty value.  The context is guaranteed to run without deadlock.

  Raises:
    ValueError: If the signature is already on the stack.
    RuntimeError: If another thread or function modifies the current stack
      entry during the yield.
  """
    stack = _get_critical_section_stack()
    if signature in stack:
        raise ValueError(f'Attempting to lock a CriticalSection (signature={signature}) in which we are already running. This is illegal and may cause deadlocks.')
    stack.append(signature)
    try:
        yield
    finally:
        received_signature = stack.pop()
        if received_signature != signature:
            raise RuntimeError(f'CriticalSection stack inconsistency: expected signature {signature} but received {received_signature}')
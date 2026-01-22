import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
def _get_wrapper(x, tf_should_use_helper):
    """Create a wrapper for object x, whose class subclasses type(x).

  The wrapper will emit a warning if it is deleted without any of its
  properties being accessed or methods being called.

  Args:
    x: The instance to wrap.
    tf_should_use_helper: The object that tracks usage.

  Returns:
    An object wrapping `x`, of type `type(x)`.
  """
    type_x = type(x)
    memoized = _WRAPPERS.get(type_x, None)
    if memoized:
        return memoized(x, tf_should_use_helper)
    tx = copy.deepcopy(ShouldUseWrapper)
    bases = getattr(tx, '__orig_bases__', tx.__bases__)

    def set_body(ns):
        ns.update(tx.__dict__)
        return ns
    copy_tx = types.new_class(tx.__name__, bases, exec_body=set_body)
    copy_tx.__init__ = _new__init__
    copy_tx.__getattribute__ = _new__getattribute__
    for op in OVERLOADABLE_OPERATORS:
        if hasattr(type_x, op):
            setattr(copy_tx, op, getattr(type_x, op))
    copy_tx.mark_used = _new_mark_used
    copy_tx.__setattr__ = _new__setattr__
    _WRAPPERS[type_x] = copy_tx
    return copy_tx(x, tf_should_use_helper)
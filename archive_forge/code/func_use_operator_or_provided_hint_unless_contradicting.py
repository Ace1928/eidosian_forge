import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def use_operator_or_provided_hint_unless_contradicting(operator, hint_attr_name, provided_hint_value, message):
    """Get combined hint in the case where operator.hint should equal hint.

  Args:
    operator:  LinearOperator that a meta-operator was initialized with.
    hint_attr_name:  String name for the attribute.
    provided_hint_value:  Bool or None. Value passed by user in initialization.
    message:  Error message to print if hints contradict.

  Returns:
    True, False, or None.

  Raises:
    ValueError: If hints contradict.
  """
    op_hint = getattr(operator, hint_attr_name)
    if op_hint is False and provided_hint_value:
        raise ValueError(message)
    if op_hint and provided_hint_value is False:
        raise ValueError(message)
    if op_hint or provided_hint_value:
        return True
    if op_hint is False or provided_hint_value is False:
        return False
    return None
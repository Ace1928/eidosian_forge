from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
@classmethod
def _overload_operator(cls, tensor_class, operator):
    """Overload an operator with the same implementation as a base Tensor class.

    We pull the operator out of the class dynamically to avoid ordering issues.

    Args:
      tensor_class: The (Composite)Tensor to get the method from.
      operator: string. The operator name.
    """
    tensor_oper = getattr(tensor_class, operator)
    tensor_oper = getattr(tensor_oper, '__func__', tensor_oper)
    setattr(cls, operator, tensor_oper)
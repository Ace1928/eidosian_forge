from tensorflow.core.protobuf import composite_tensor_variant_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_composite_tensor_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
Returns the `ExtensionType` value encoded by a variant scalar tensor.

  Args:
    encoded: A Tensor returned by `composite_tensor_to_variants`.
    type_spec: The `TypeSpec` of the original value.  This is used to determine
      the number and types of the component tensors that comprise the decoded
      value.  Must be compatible with the `TypeSpec` serilized in `encoded`.
    name: Optional name for the operation.

  Returns:
    An `ExtensionType` value that is compatible with `TypeSpec`.

  Raises:
    TypeError: If `encoded` is not a Tensor with dtype=variant.
    InvalidArgumentError: If `encoded` is not compatible with `type_spec`.
  
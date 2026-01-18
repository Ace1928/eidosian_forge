from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Load an Op from a SignatureDef created by op_signature_def().

  Args:
    signature_def: a SignatureDef proto
    key: string key to op in the SignatureDef outputs.
    import_scope: Scope used to import the op

  Returns:
    Op (or possibly Tensor) in the graph with the same name as saved in the
      SignatureDef.

  Raises:
    NotFoundError: If the op could not be found in the graph.
  
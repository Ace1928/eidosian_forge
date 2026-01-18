import collections
from typing import Dict, List, Union
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export
Saves name based Tensor into a Checkpoint.

  The function prepares the input dictionary to the format of a `sharded_save`,
  so that it can take advantage of DTensor SPMD based distributed save.

  Same as restore, the function only supports saving on the single mesh.

  Args:
    mesh: The single mesh that all Tensors would be restored to.
    checkpoint_prefix : The prefix of checkpoint to be restored.
    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The
      DTensor shape/dtype must match the tensors being saved/restored for now.
  
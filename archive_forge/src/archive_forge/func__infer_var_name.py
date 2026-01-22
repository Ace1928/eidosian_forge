import collections
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export
def _infer_var_name(var):
    """Returns name of the `var`.

  Args:
    var: A list. The list can contain either of the following:
      (i) A single `Variable`
      (ii) A single `ResourceVariable`
      (iii) Multiple `Variable` objects which must be slices of the same larger
        variable.
      (iv) A single `PartitionedVariable`

  Returns:
    Name of the `var`
  """
    name_to_var_dict = saveable_object_util.op_list_to_dict(var)
    if len(name_to_var_dict) > 1:
        raise TypeError('`var` = %s passed as arg violates the constraints.  name_to_var_dict = %s' % (var, name_to_var_dict))
    return list(name_to_var_dict.keys())[0]
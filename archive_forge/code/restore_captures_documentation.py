import warnings
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import resource
Restore captures for the concrete function.

  Used at deserialization time.  For functions that are being deserialized,
  saved model restores objects that tensors were captured from, but functions
  only know about their tensors -- object information is destroyed by tracing.
  This additional logic extracts the tensors which the function originally
  captured.

  Args:
    concrete_function: the concrete function for which to restore captures
    inputs: a list tensors or other Python objects (such as variables) which
      contain tensors that were originally captured by the function
  
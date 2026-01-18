import warnings
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import function_utils
from tensorflow.python.util import variable_utils
def wrapper_helper(*args):
    """Wrapper for passing nested structures to and from tf.data functions."""
    nested_args = structure.from_compatible_tensor_list(self._input_structure, args)
    if not _should_unpack(nested_args):
        nested_args = (nested_args,)
    ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    ret = variable_utils.convert_variables_to_tensors(ret)
    if _should_pack(ret):
        ret = tuple(ret)
    try:
        self._output_structure = structure.type_spec_from_value(ret)
    except (ValueError, TypeError) as e:
        raise TypeError(f'Unsupported return value from function passed to {transformation_name}: {ret}.') from e
    return ret
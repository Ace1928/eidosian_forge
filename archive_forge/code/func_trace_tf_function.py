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
def trace_tf_function(defun_kwargs):

    def wrapped_fn(*args):
        ret = wrapper_helper(*args)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]
    func_name = defun_kwargs.pop('func_name', 'wrapped_fn')
    tf_function = def_function.Function(python_function=wrapped_fn, name=func_name, input_signature=structure.get_flat_tensor_specs(self._input_structure), autograph=False, experimental_attributes=defun_kwargs)
    return tf_function.get_concrete_function
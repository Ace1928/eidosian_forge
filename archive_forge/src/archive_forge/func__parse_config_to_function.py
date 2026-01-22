import hashlib
import numbers
import sys
import types as python_types
import warnings
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
def _parse_config_to_function(config, custom_objects, func_attr_name, func_type_attr_name, module_attr_name):
    """Reconstruct the function from the config."""
    globs = globals()
    module = config.pop(module_attr_name, None)
    if module in sys.modules:
        globs.update(sys.modules[module].__dict__)
    elif module is not None:
        warnings.warn('{} is not loaded, but a layer uses it. It may cause errors.'.format(module), UserWarning)
    if custom_objects:
        globs.update(custom_objects)
    function_type = config.pop(func_type_attr_name)
    if function_type == 'function':
        function = generic_utils.deserialize_keras_object(config[func_attr_name], custom_objects=custom_objects, printable_module_name='function in wrapper')
    elif function_type == 'lambda':
        function = generic_utils.func_load(config[func_attr_name], globs=globs)
    else:
        raise TypeError('Unknown function type:', function_type)
    return function
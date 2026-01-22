import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
@classmethod
def _parse_function_from_config(cls, config, custom_objects, func_attr_name, module_attr_name, func_type_attr_name):
    globs = globals().copy()
    module = config.pop(module_attr_name, None)
    if module in sys.modules:
        globs.update(sys.modules[module].__dict__)
    elif module is not None:
        warnings.warn('{} is not loaded, but a Lambda layer uses it. It may cause errors.'.format(module), UserWarning)
    if custom_objects:
        globs.update(custom_objects)
    function_type = config.pop(func_type_attr_name)
    if function_type == 'function':
        function = generic_utils.deserialize_keras_object(config[func_attr_name], custom_objects=custom_objects, printable_module_name='function in Lambda layer')
    elif function_type == 'lambda':
        function = generic_utils.func_load(config[func_attr_name], globs=globs)
    elif function_type == 'raw':
        function = config[func_attr_name]
    else:
        raise TypeError('Unknown function type:', function_type)
    return function
from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def _normalize_outputs(outputs, function_name, signature_key):
    """Normalize outputs if necessary and check that they are tensors."""
    if not isinstance(outputs, collections_abc.Mapping):
        if hasattr(outputs, '_asdict'):
            outputs = outputs._asdict()
        else:
            if not isinstance(outputs, collections_abc.Sequence):
                outputs = [outputs]
            outputs = {'output_{}'.format(output_index): output for output_index, output in enumerate(outputs)}
    for key, value in outputs.items():
        if not isinstance(key, compat.bytes_or_text_types):
            raise ValueError(f'Got a dictionary with a non-string key {key!r} in the output of the function {compat.as_str_any(function_name)} used to generate the SavedModel signature {signature_key!r}.')
        if not isinstance(value, (tensor.Tensor, composite_tensor.CompositeTensor)):
            raise ValueError(f'Got a non-Tensor value {value!r} for key {key!r} in the output of the function {compat.as_str_any(function_name)} used to generate the SavedModel signature {signature_key!r}. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.')
    return outputs
import abc
import typing
import warnings
import typing_extensions
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _build_extension_type_constructor(cls):
    """Builds a constructor for tf.ExtensionType subclass `cls`."""
    fields = cls._tf_extension_type_fields()
    got_default = False
    keyword_only_start = len(fields)
    for i in range(len(fields)):
        if got_default:
            if fields[i].default is _NO_DEFAULT:
                keyword_only_start = i
                break
        elif fields[i].default is not _NO_DEFAULT:
            got_default = True
    params = []
    for i, field in enumerate(fields):
        if i < keyword_only_start:
            kind = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
        else:
            kind = tf_inspect.Parameter.KEYWORD_ONLY
        if field.default is _NO_DEFAULT:
            default = tf_inspect.Parameter.empty
        else:
            default = field.default
        params.append(tf_inspect.Parameter(field.name, kind, default=default, annotation=field.value_type))
    signature = tf_inspect.Signature(params, return_annotation=cls.__name__)

    def __init__(self, *args, **kwargs):
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        self.__dict__.update(bound_args.arguments)
        self._tf_extension_type_convert_fields()
        self.__validate__()
    __init__.__signature__ = tf_inspect.Signature([tf_inspect.Parameter('self', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD)] + params, return_annotation=cls)
    cls.__init__ = __init__
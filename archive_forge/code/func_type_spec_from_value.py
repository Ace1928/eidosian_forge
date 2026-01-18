import collections
import functools
import itertools
import wrapt
from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.nest_util import CustomNestProtocol
from tensorflow.python.util.tf_export import tf_export
def type_spec_from_value(element, use_fallback=True):
    """Creates a type specification for the given value.

  Args:
    element: The element to create the type specification for.
    use_fallback: Whether to fall back to converting the element to a tensor
      in order to compute its `TypeSpec`.

  Returns:
    A nested structure of `TypeSpec`s that represents the type specification
    of `element`.

  Raises:
    TypeError: If a `TypeSpec` cannot be built for `element`, because its type
      is not supported.
  """
    spec = type_spec._type_spec_from_value(element)
    if spec is not None:
        return spec
    if isinstance(element, collections_abc.Mapping):
        if isinstance(element, collections.defaultdict):
            ctor = lambda items: type(element)(element.default_factory, items)
        else:
            ctor = type(element)
        return ctor([(k, type_spec_from_value(v)) for k, v in element.items()])
    if isinstance(element, tuple):
        if hasattr(element, '_fields') and isinstance(element._fields, collections_abc.Sequence) and all((isinstance(f, str) for f in element._fields)):
            if isinstance(element, wrapt.ObjectProxy):
                element_type = type(element.__wrapped__)
            else:
                element_type = type(element)
            return element_type(*[type_spec_from_value(v) for v in element])
        return tuple([type_spec_from_value(v) for v in element])
    if hasattr(element.__class__, '__attrs_attrs__'):
        attrs = getattr(element.__class__, '__attrs_attrs__')
        return type(element)(*[type_spec_from_value(getattr(element, a.name)) for a in attrs])
    if isinstance(element, CustomNestProtocol):
        metadata, children = element.__tf_flatten__()
        return element.__tf_unflatten__(metadata, type_spec_from_value(children))
    if use_fallback:
        try:
            tensor = ops.convert_to_tensor(element)
            spec = type_spec_from_value(tensor)
            if spec is not None:
                return spec
        except (ValueError, TypeError) as e:
            logging.vlog(3, 'Failed to convert %r to tensor: %s' % (type(element).__name__, e))
    raise TypeError('Could not build a `TypeSpec` for {} with type {}'.format(element, type(element).__name__))
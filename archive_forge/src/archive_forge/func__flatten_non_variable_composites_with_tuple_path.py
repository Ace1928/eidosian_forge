import re
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
def _flatten_non_variable_composites_with_tuple_path(structure, path_prefix=()):
    """Flattens composite tensors with tuple path expect variables."""
    for path, child in nest.flatten_with_tuple_paths(structure):
        if isinstance(child, composite_tensor.CompositeTensor) and (not _is_variable(child)):
            spec = child._type_spec
            yield from _flatten_non_variable_composites_with_tuple_path(spec._to_components(child), path_prefix + path + (spec.value_type.__name__,))
        else:
            yield (path_prefix + path, child)
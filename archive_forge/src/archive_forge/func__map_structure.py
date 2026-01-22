import collections
import functools
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _map_structure(pyobj, coders):
    for can, do in reversed(coders):
        if can(pyobj):
            recursion_fn = functools.partial(_map_structure, coders=coders)
            return do(pyobj, recursion_fn)
    raise NotEncodableError(f'No encoder for object {str(pyobj)} of type {type(pyobj)}.')
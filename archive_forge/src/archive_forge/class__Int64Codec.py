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
class _Int64Codec:
    """Codec for Python integers (limited to 64 bit values)."""

    def can_encode(self, pyobj):
        return not isinstance(pyobj, bool) and isinstance(pyobj, int)

    def do_encode(self, int_value, encode_fn):
        del encode_fn
        value = struct_pb2.StructuredValue()
        value.int64_value = int_value
        return value

    def can_decode(self, value):
        return value.HasField('int64_value')

    def do_decode(self, value, decode_fn):
        del decode_fn
        return int(value.int64_value)
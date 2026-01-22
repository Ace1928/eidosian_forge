from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes
Fix endiness of tensor contents.

  Args:
    graph_def: Target graph_def to change endiness.
    from_endiness: The original endianness format. "big" or "little"
    to_endiness: The target endianness format. "big" or "little"
  
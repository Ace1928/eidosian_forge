import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def _create_tuple_proto(self, num_outputs):
    shardings = [xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED)] * num_outputs
    return xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=shardings)
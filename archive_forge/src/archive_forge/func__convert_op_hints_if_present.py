import copy
import datetime
import sys
from absl import logging
import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph
def _convert_op_hints_if_present(sess, graph_def, output_tensors, hinted_outputs_nodes):
    if is_frozen_graph(sess):
        raise ValueError('Try to convert op hints, needs unfrozen graph.')
    output_arrays = [get_tensor_name(tensor) for tensor in output_tensors]
    graph_def = _convert_to_constants.convert_variables_to_constants(sess, graph_def, output_arrays + hinted_outputs_nodes)
    graph_def = convert_op_hints_to_stubs(graph_def=graph_def)
    return graph_def
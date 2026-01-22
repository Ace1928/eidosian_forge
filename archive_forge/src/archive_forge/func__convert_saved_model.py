import collections
from functools import partial  # pylint: disable=g-importing-member
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _convert_saved_model(self):
    """Convert the input SavedModel."""
    graph = ops.Graph()
    with session.Session(graph=graph) as sess:
        input_meta_graph_def = loader.load(sess, self._input_saved_model_tags, self._input_saved_model_dir)
        input_signature_def = input_meta_graph_def.signature_def[self._input_saved_model_signature_key]

        def _gather_names(tensor_info):
            """Get the node names from a TensorInfo."""
            return {tensor_info[key].name.split(':')[0] for key in tensor_info}
        output_node_names = _gather_names(input_signature_def.inputs).union(_gather_names(input_signature_def.outputs))
        for collection_key in self._collections_to_keep(input_meta_graph_def.collection_def):
            for op in sess.graph.get_collection(collection_key):
                if isinstance(op, ops.Operation):
                    output_node_names.add(op.name.split(':')[0])
        frozen_graph_def = convert_to_constants.convert_variables_to_constants(sess, sess.graph.as_graph_def(add_shapes=True), list(output_node_names))
        self._grappler_meta_graph_def = meta_graph_pb2.MetaGraphDef()
        self._grappler_meta_graph_def.graph_def.CopyFrom(frozen_graph_def)
        for collection_key in self._collections_to_keep(input_meta_graph_def.collection_def):
            self._grappler_meta_graph_def.collection_def[collection_key].CopyFrom(input_meta_graph_def.collection_def[collection_key])
        self._add_nodes_denylist()
        self._grappler_meta_graph_def.meta_info_def.CopyFrom(input_meta_graph_def.meta_info_def)
        self._grappler_meta_graph_def.signature_def[self._input_saved_model_signature_key].CopyFrom(input_signature_def)
    self._run_conversion()
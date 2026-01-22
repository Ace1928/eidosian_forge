import collections
import copy
import functools
import itertools
import threading
import warnings
import weakref
import numpy as np
from google.protobuf import json_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.tools.docs import doc_controls
class TensorFlowOpLayer(Layer):
    """Wraps a TensorFlow Operation in a Layer.

  This class is used internally by the Functional API. When a user
  uses a raw TensorFlow Operation on symbolic tensors originating
  from an `Input` Layer, the resultant operation will be wrapped
  with this Layer object in order to make the operation compatible
  with the Keras API.

  This Layer will create a new, identical operation (except for inputs
  and outputs) every time it is called. If `run_eagerly` is `True`,
  the op creation and calculation will happen inside an Eager function.

  Instances of this Layer are created when `autolambda` is called, which
  is whenever a Layer's `__call__` encounters symbolic inputs that do
  not have Keras metadata, or when a Network's `__init__` encounters
  outputs that do not have Keras metadata.

  Attributes:
    node_def: String, the serialized NodeDef of the Op this layer will wrap.
    name: String, the name of the Layer.
    constants: Dict of NumPy arrays, the values of any Tensors needed for this
      Operation that do not originate from a Keras `Input` Layer. Since all
      placeholders must come from Keras `Input` Layers, these Tensors must be
      treated as constant in the Functional API.
    trainable: Bool, whether this Layer is trainable. Currently Variables are
      not supported, and so this parameter has no effect.
    dtype: The default dtype of this Layer. Inherited from `Layer` and has no
      effect on this class, however is used in `get_config`.
  """

    @trackable.no_automatic_dependency_tracking
    def __init__(self, node_def, name, constants=None, trainable=True, dtype=None):
        super(TensorFlowOpLayer, self).__init__(name=_TF_OP_LAYER_NAME_PREFIX + name, trainable=trainable, dtype=dtype, autocast=False)
        if isinstance(node_def, dict):
            self.node_def = json_format.ParseDict(node_def, node_def_pb2.NodeDef())
        else:
            if not isinstance(node_def, bytes):
                node_def = node_def.encode('utf-8')
            self.node_def = node_def_pb2.NodeDef.FromString(node_def)
        self.constants = {int(index): constant for index, constant in constants.items()} if constants is not None else {}
        self.built = True
        self._must_restore_from_config = True

    def call(self, inputs):
        if context.executing_eagerly():
            return self._defun_call(inputs)
        return self._make_op(inputs)

    def _make_node_def(self, graph):
        node_def = node_def_pb2.NodeDef()
        node_def.CopyFrom(self.node_def)
        node_def.attr['_cloned'].b = True
        node_def.name = graph.unique_name(node_def.name)
        return node_def

    def _make_op(self, inputs):
        inputs = nest.flatten(inputs)
        graph = inputs[0].graph
        node_def = self._make_node_def(graph)
        with graph.as_default():
            for index, constant in self.constants.items():
                value = tensor_util.constant_value(constant)
                if value is not None:
                    constant = constant_op.constant(value, name=node_def.input[index])
                inputs.insert(index, constant)
            c_op = ops._create_c_op(graph, node_def, inputs, control_inputs=[])
            op = graph._create_op_from_tf_operation(c_op)
            op._control_flow_post_processing()
            op_type = compat.as_str(op.op_def.name)
            attr_names = [compat.as_str(attr.name) for attr in op.op_def.attr]
            attrs = []
            for attr_name in attr_names:
                attrs.append(attr_name)
                attrs.append(op.get_attr(attr_name))
            attrs = tuple(attrs)
            backprop.record_gradient(op_type, op.inputs, attrs, op.outputs)
            if len(op.outputs) == 1:
                return op.outputs[0]
            return op.outputs

    @def_function.function
    def _defun_call(self, inputs):
        """Wraps the op creation method in an Eager function for `run_eagerly`."""
        return self._make_op(inputs)

    def get_config(self):
        config = super(TensorFlowOpLayer, self).get_config()
        config.update({'name': config['name'][len(_TF_OP_LAYER_NAME_PREFIX):], 'node_def': json_format.MessageToDict(self.node_def), 'constants': {i: backend.get_value(c) for i, c in self.constants.items()}})
        return config
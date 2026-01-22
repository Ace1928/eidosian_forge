import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
def _create_keras_history_helper(tensors, processed_ops, created_layers):
    """Helper method for `create_keras_history`.

  Args:
    tensors: A structure of Tensors for which to create Keras metadata.
    processed_ops: Set. TensorFlow operations that have already been wrapped in
      `TensorFlowOpLayer` instances.
    created_layers: List. The `TensorFlowOpLayer` instances created.

  Returns:
    Tuple. First element is the updated set of TensorFlow Operations that
    have been wrapped in `TensorFlowOpLayer` instances. Second element is
    a list of the `TensorFlowOpLayer` instances created.
  """
    if ops.executing_eagerly_outside_functions():
        raise ValueError('`create_keras_history` should only be called if eager is disabled!')
    from tensorflow.python.keras.engine import base_layer
    tensor_list = nest.flatten(tensors)
    sparse_ops = []
    ragged_tensors = []
    for tensor in tensor_list:
        if getattr(tensor, '_keras_history', None) is not None:
            continue
        if isinstance(tensor, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
            sparse_ops.append(tensor.op)
            continue
        if tf_utils.is_ragged(tensor):
            ragged_tensors.append(tensor)
            continue
        op = tensor.op
        if op not in processed_ops:
            op_inputs = list(op.inputs)
            constants = {}
            layer_inputs = []
            for i, op_input in enumerate(op_inputs):
                if uses_keras_history(op_input):
                    layer_inputs.append(op_input)
                else:
                    ds_with_session = distribute_lib.in_cross_replica_context() and (not ops.executing_eagerly_outside_functions())
                    using_xla = control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())
                    if ds_with_session or using_xla or _UNSAFE_GRAPH_OP_LAYER_CREATION:
                        constants[i] = op_input
                    else:
                        with ops.init_scope():
                            constants[i] = backend.function([], op_input)([])
            layer_inputs = unnest_if_single_tensor(layer_inputs)
            processed_ops, created_layers = _create_keras_history_helper(layer_inputs, processed_ops, created_layers)
            name = op.name
            node_def = op.node_def.SerializeToString()
            op_layer = base_layer.TensorFlowOpLayer(node_def, constants=constants, name=name)
            created_layers.append(op_layer)
            op_layer._set_connectivity_metadata(args=(layer_inputs,), kwargs={}, outputs=op.outputs)
            processed_ops.update([op])
    if sparse_ops or ragged_tensors:
        lambda_example = '\n    weights_mult = lambda x: tf.sparse.sparse_dense_matmul(x, weights)\n    output = tf.keras.layers.Lambda(weights_mult)(input)\n    '
        raise ValueError('Tensorflow ops that generate ragged or sparse tensor outputs are currently not supported by Keras automatic op wrapping. Please wrap these ops in a Lambda layer: \n\n```\n{example}\n```\nSparse ops encountered: {sparse_ops}\nRagged tensors encountered: {ragged_tensors}\n'.format(example=lambda_example, sparse_ops=str(sparse_ops), ragged_tensors=str(ragged_tensors)))
    return (processed_ops, created_layers)
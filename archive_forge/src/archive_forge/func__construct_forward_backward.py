import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _construct_forward_backward(self, num_doutputs):
    """Constructs a pair of forward and backward functions.

    Args:
      num_doutputs: The constructed backprop function will take output gradients
        for the first `num_doutputs` outputs of the forward function. Defaults
        to the number of outputs for the inference function, but when
        higher-order gradients are computed this will increase to include side
        outputs.

    Returns:
      A pair of (forward_function, backward_function):
        forward_function: A re-generated inference function (an
          AtomicFunction) to account for new side outputs, if any extra
          were required when building the backward pass.
        backward_function: A ConcreteFunction that Takes `num_doutputs`
          arguments and returns gradients with respect to inputs of the forward
          function.
    """
    trainable_outputs = [output for output in self._func_graph.outputs[:num_doutputs] if backprop_util.IsTrainable(output)]
    signature = []
    for t in trainable_outputs:
        signature.append(tensor_lib.TensorSpec(*default_gradient.shape_and_dtype(t)))

    def _backprop_function(*grad_ys):
        with ops.device(None):
            return gradients_util._GradientsHelper(trainable_outputs, self._func_graph.inputs, grad_ys=grad_ys, src_graph=self._func_graph)
    with self._func_graph.as_default():
        backwards_graph = func_graph_module.FuncGraph(_backward_name(self._func_graph.name))
        func_graph_module.func_graph_from_py_func(name=backwards_graph.name, python_func=_backprop_function, args=[], kwargs={}, signature=signature, func_graph=backwards_graph)
        backwards_graph_captures = backwards_graph.external_captures
        captures_from_forward = [c for c in backwards_graph_captures if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]
        existing_outputs = object_identity.ObjectIdentitySet(self._func_graph.outputs)
        for capture in captures_from_forward:
            if capture not in existing_outputs:
                existing_outputs.add(capture)
                self._func_graph.outputs.append(capture)
        forward_function, backward_function = _create_forward_backward_with_graph(self._attrs, self._func_graph, backwards_graph)
        return (forward_function, backward_function)
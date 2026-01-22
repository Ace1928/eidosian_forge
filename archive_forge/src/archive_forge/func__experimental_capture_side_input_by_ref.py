import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _experimental_capture_side_input_by_ref(self, identifier: Hashable, func: Callable[[], Any]) -> ...:
    """Implement capturing side input by reference for tf.function.

    Note that this API will only register the capture in the func_graph where
    it is called. In the case of nested graph, like nested tf.function or
    tf.while, the outer graph is not aware of this capture in the inner graph.
    Thus, the outer tf.function will not retrace when the by-ref capture
    changes. It's the user's responsibility to call this API in the outer
    func_graph as well if proper retracing is needed.

    For example:

    ```
    x = 1

    # Correct usage
    @tf.function
    def f_1():
      graph = tf.compat.v1.get_default_graph()
      # Capture the same x for the outer tf.function
      graph._experimental_capture_side_input_by_ref("x", lambda: x)

      @tf.function
      def g():
        graph = tf.compat.v1.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
        return cap_x + 1

      return g()

    # Incorrect usage
    @tf.function
    def f_2():

      @tf.function
      def g():
        graph = tf.compat.v1.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
        return cap_x + 1

      return g()

    assert f_1() == 2
    assert f_2() == 2
    x = 2
    assert f_1() == 3
    assert f_2() == 2  # This is incorrect
    ```

    Args:
      identifier: A hashable object as the key for the capture.
      func: A Python function that takes no arguments and returns the value of
        side input. The function is evaluated at function call time.

    Returns:
      A nested structure with the same structure as the side input. Tensors
        are replaced with placehoders, and non-tensors remain the same.

    """
    if context.executing_eagerly():
        return func()

    def maybe_convert_to_tensor():
        value = func()
        if not (isinstance(value, core.Value) or isinstance(value, core.Symbol)):
            value = constant_op.constant(value)
        return value
    placeholder = self._function_captures._capture_by_ref(self, maybe_convert_to_tensor, identifier)
    return placeholder
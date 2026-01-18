import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
def watch_gradients_by_tensors(self, graph, tensors):
    """Watch gradient tensors by x-tensor(s).

    The side effect of this method is that when gradient tensor(s) are created
    with respect to the any paths that include the `x_tensor`s, the gradient
    tensor(s) with respect to the tensor will be registered with this
    this `GradientsDebugger` instance and can later be retrieved, with the
    methods `gradient_tensor` and `gradient_tensors`.

    Unlike the method `identify_gradient`, this method is used to retrieve
    gradient tensors after the construction of the forward subgraph has
    completed (but before the construction of the backward subgraph).

    This method is the same as `watch_gradients_by_x_tensor_names` except that
    the tensors are specified by the Python `tf.Tensor` or `tf.Variable`
    objects, instead by name patterns.

    Example:

    ```python
    x = tf.Variable(1.0)
    y = tf.add(x, x, name="y")
    z = tf.square(debug_y)

    # Create a train op under the grad_debugger context.
    grad_debugger = tf_debug.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensors(y):
      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)

    # Now we can reflect through grad_debugger to get the gradient tensor
    # with respect to y.
    y_grad = grad_debugger.gradient_tensor(y)
    # or
    y_grad = grad_debugger.gradient_tensor("y:0")
    ```

    Args:
      graph: the `tf.Graph` to watch the gradients on.
      tensors: a `tf.Tensor` or `tf.Variable` object, or a list of such objects.

    Returns:
      The GradientsDebugger instance itself.
    """
    if not isinstance(tensors, list):
        tensors = [tensors]
    tensor_name_regex = []
    for tensor in tensors:
        tensor_name_regex.append(re.escape(tensor.name) + '$')
    tensor_name_regex = '(' + '|'.join(tensor_name_regex) + ')'
    return self.watch_gradients_by_tensor_names(graph, tensor_name_regex)
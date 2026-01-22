from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import func_graph
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class _EstimatorSpecFunction(tf.compat.v2.__internal__.function.Function):
    """Wraps graph functions defined for a function returning an EstimatorSpec.

  This object handles creation of the graph functions.
  """

    def __init__(self, python_function, name, variable_holder=None, **kwargs):
        super(_EstimatorSpecFunction, self).__init__(python_function, name, **kwargs)
        self._variable_holder = variable_holder

    def _create_graph_function(self, args, kwargs, **other_kwargs):
        _ = other_kwargs
        wrapped_graph = _EstimatorWrappedGraph(self._variable_holder)
        return wrapped_graph.wrap_model_fn(self._python_function, self._name, signature=self.input_signature, args=args, kwargs=kwargs)
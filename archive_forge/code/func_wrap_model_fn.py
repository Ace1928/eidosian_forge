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
def wrap_model_fn(self, model_fn, mode, args=None, kwargs=None, signature=None):
    """Wraps a model function, and stores the returned estimator spec."""
    if self._concrete_model_fn is not None:
        raise ValueError('`wrap_model_fn` should be only called once per graph.')

    def fn(*args, **kwargs):
        """Returns tensor and op outputs from the returned spec."""
        ret = model_fn(*args, **kwargs)
        if isinstance(ret, model_fn_lib.EstimatorSpec):
            self._estimator_spec = ret
            return _filter_estimator_spec_outputs(ret)
        return ret
    name = 'model_fn_{}'.format(mode)
    self._concrete_model_fn = self._wrap_function(fn, args, kwargs, signature, name)
    return self._concrete_model_fn
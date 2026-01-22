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
def add_mode(self, fn, mode, input_signature=None):
    if mode in self._functions:
        raise ValueError('ModelFunction object has multiple functions with name {}.'.format(mode))
    spec_fn = EstimatorSpecFunction(fn, mode, config=self._config, params=self._params, variable_holder=self._variable_holder, input_signature=input_signature)
    self._functions[mode] = spec_fn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
def _assert_checkpoint(self, n_classes, expected_global_step, expected_age_weight=None, expected_bias=None):
    logits_dimension = n_classes if n_classes > 2 else 1
    shapes = {name: shape for name, shape in tf.train.list_variables(self._model_dir)}
    self.assertEqual([], shapes[tf.compat.v1.GraphKeys.GLOBAL_STEP])
    self.assertEqual(expected_global_step, tf.train.load_variable(self._model_dir, tf.compat.v1.GraphKeys.GLOBAL_STEP))
    self.assertEqual([1, logits_dimension], shapes[AGE_WEIGHT_NAME])
    if expected_age_weight is not None:
        self.assertAllEqual(expected_age_weight, tf.train.load_variable(self._model_dir, AGE_WEIGHT_NAME))
    self.assertEqual([logits_dimension], shapes[BIAS_NAME])
    if expected_bias is not None:
        self.assertAllEqual(expected_bias, tf.train.load_variable(self._model_dir, BIAS_NAME))
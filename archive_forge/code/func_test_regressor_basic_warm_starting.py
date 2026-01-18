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
def test_regressor_basic_warm_starting(self):
    """Tests correctness of LinearRegressor default warm-start."""
    age = self._fc_lib.numeric_column('age')
    linear_regressor = self._linear_regressor_fn(feature_columns=[age], model_dir=self._ckpt_and_vocab_dir, optimizer='SGD')
    linear_regressor.train(input_fn=self._input_fn, max_steps=1)
    warm_started_linear_regressor = self._linear_regressor_fn(feature_columns=[age], optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=linear_regressor.model_dir)
    warm_started_linear_regressor.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_linear_regressor.get_variable_names():
        self.assertAllClose(linear_regressor.get_variable_value(variable_name), warm_started_linear_regressor.get_variable_value(variable_name))
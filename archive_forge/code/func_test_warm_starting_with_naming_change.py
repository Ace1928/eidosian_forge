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
def test_warm_starting_with_naming_change(self):
    """Tests warm-starting with a Tensor name remapping."""
    age_in_years = self._fc_lib.numeric_column('age_in_years')
    linear_classifier = self._linear_classifier_fn(feature_columns=[age_in_years], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
    linear_classifier.train(input_fn=self._input_fn, max_steps=1)
    warm_started_linear_classifier = self._linear_classifier_fn(feature_columns=[self._fc_lib.numeric_column('age')], n_classes=4, optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=linear_classifier.model_dir, var_name_to_prev_var_name={AGE_WEIGHT_NAME: AGE_WEIGHT_NAME.replace('age', 'age_in_years')}))
    warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
    self.assertAllClose(linear_classifier.get_variable_value(AGE_WEIGHT_NAME.replace('age', 'age_in_years')), warm_started_linear_classifier.get_variable_value(AGE_WEIGHT_NAME))
    self.assertAllClose(linear_classifier.get_variable_value(BIAS_NAME), warm_started_linear_classifier.get_variable_value(BIAS_NAME))
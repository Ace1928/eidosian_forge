from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class BaseDNNRegressorPredictTest(object):

    def __init__(self, dnn_regressor_fn, fc_impl=feature_column):
        self._dnn_regressor_fn = dnn_regressor_fn
        self._fc_impl = fc_impl

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def test_one_dim(self):
        """Asserts predictions for one-dimensional input and logits."""
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), global_step=0, model_dir=self._model_dir)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=(2, 2), feature_columns=(self._fc_impl.numeric_column('x'),), model_dir=self._model_dir)
        input_fn = numpy_io.numpy_input_fn(x={'x': np.array([[10.0]])}, batch_size=1, shuffle=False)
        self.assertAllClose({prediction_keys.PredictionKeys.PREDICTIONS: [-2.08]}, next(dnn_regressor.predict(input_fn=input_fn)))

    def test_multi_dim(self):
        """Asserts predictions for multi-dimensional input and logits."""
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), 100, self._model_dir)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=(2, 2), feature_columns=(self._fc_impl.numeric_column('x', shape=(2,)),), label_dimension=3, model_dir=self._model_dir)
        input_fn = numpy_io.numpy_input_fn(x={'x': np.array([[10.0, 8.0]])}, batch_size=1, shuffle=False)
        self.assertAllClose({prediction_keys.PredictionKeys.PREDICTIONS: [-0.48, 0.48, 0.39]}, next(dnn_regressor.predict(input_fn=input_fn)))
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
class BaseLinearRegressorEvaluationTest(object):

    def __init__(self, linear_regressor_fn, fc_lib=feature_column):
        self._linear_regressor_fn = linear_regressor_fn
        self._fc_lib = fc_lib

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def test_evaluation_for_simple_data(self):
        with tf.Graph().as_default():
            tf.Variable([[11.0]], name=AGE_WEIGHT_NAME)
            tf.Variable([2.0], name=BIAS_NAME)
            tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir)
        eval_metrics = linear_regressor.evaluate(input_fn=lambda: ({'age': ((1,),)}, ((10.0,),)), steps=1)
        self.assertDictEqual({metric_keys.MetricKeys.LOSS: 9.0, metric_keys.MetricKeys.LOSS_MEAN: 9.0, metric_keys.MetricKeys.PREDICTION_MEAN: 13.0, metric_keys.MetricKeys.LABEL_MEAN: 10.0, tf.compat.v1.GraphKeys.GLOBAL_STEP: 100}, eval_metrics)

    def test_evaluation_batch(self):
        """Tests evaluation for batch_size==2."""
        with tf.Graph().as_default():
            tf.Variable([[11.0]], name=AGE_WEIGHT_NAME)
            tf.Variable([2.0], name=BIAS_NAME)
            tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir)
        eval_metrics = linear_regressor.evaluate(input_fn=lambda: ({'age': ((1,), (1,))}, ((10.0,), (10.0,))), steps=1)
        self.assertDictEqual({metric_keys.MetricKeys.LOSS: 18.0, metric_keys.MetricKeys.LOSS_MEAN: 9.0, metric_keys.MetricKeys.PREDICTION_MEAN: 13.0, metric_keys.MetricKeys.LABEL_MEAN: 10.0, tf.compat.v1.GraphKeys.GLOBAL_STEP: 100}, eval_metrics)

    def test_evaluation_weights(self):
        """Tests evaluation with weights."""
        with tf.Graph().as_default():
            tf.Variable([[11.0]], name=AGE_WEIGHT_NAME)
            tf.Variable([2.0], name=BIAS_NAME)
            tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)

        def _input_fn():
            features = {'age': ((1,), (1,)), 'weights': ((1.0,), (2.0,))}
            labels = ((10.0,), (10.0,))
            return (features, labels)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), weight_column='weights', model_dir=self._model_dir)
        eval_metrics = linear_regressor.evaluate(input_fn=_input_fn, steps=1)
        self.assertDictEqual({metric_keys.MetricKeys.LOSS: 27.0, metric_keys.MetricKeys.LOSS_MEAN: 9.0, metric_keys.MetricKeys.PREDICTION_MEAN: 13.0, metric_keys.MetricKeys.LABEL_MEAN: 10.0, tf.compat.v1.GraphKeys.GLOBAL_STEP: 100}, eval_metrics)

    def test_evaluation_for_multi_dimensions(self):
        x_dim = 3
        label_dim = 2
        with tf.Graph().as_default():
            tf.Variable([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name=AGE_WEIGHT_NAME)
            tf.Variable([7.0, 8.0], name=BIAS_NAME)
            tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age', shape=(x_dim,)),), label_dimension=label_dim, model_dir=self._model_dir)
        input_fn = numpy_io.numpy_input_fn(x={'age': np.array([[2.0, 4.0, 5.0]])}, y=np.array([[46.0, 58.0]]), batch_size=1, num_epochs=None, shuffle=False)
        eval_metrics = linear_regressor.evaluate(input_fn=input_fn, steps=1)
        self.assertItemsEqual((metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN, metric_keys.MetricKeys.PREDICTION_MEAN, metric_keys.MetricKeys.LABEL_MEAN, tf.compat.v1.GraphKeys.GLOBAL_STEP), eval_metrics.keys())
        self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])

    def test_evaluation_for_multiple_feature_columns(self):
        with tf.Graph().as_default():
            tf.Variable([[10.0]], name=AGE_WEIGHT_NAME)
            tf.Variable([[2.0]], name=HEIGHT_WEIGHT_NAME)
            tf.Variable([5.0], name=BIAS_NAME)
            tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        batch_size = 2
        feature_columns = [self._fc_lib.numeric_column('age'), self._fc_lib.numeric_column('height')]
        input_fn = numpy_io.numpy_input_fn(x={'age': np.array([20, 40]), 'height': np.array([4, 8])}, y=np.array([[213.0], [421.0]]), batch_size=batch_size, num_epochs=None, shuffle=False)
        est = self._linear_regressor_fn(feature_columns=feature_columns, model_dir=self._model_dir)
        eval_metrics = est.evaluate(input_fn=input_fn, steps=1)
        self.assertItemsEqual((metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN, metric_keys.MetricKeys.PREDICTION_MEAN, metric_keys.MetricKeys.LABEL_MEAN, tf.compat.v1.GraphKeys.GLOBAL_STEP), eval_metrics.keys())
        self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])

    def test_evaluation_for_multiple_feature_columns_mix(self):
        with tf.Graph().as_default():
            tf.Variable([[10.0]], name=AGE_WEIGHT_NAME)
            tf.Variable([[2.0]], name=HEIGHT_WEIGHT_NAME)
            tf.Variable([5.0], name=BIAS_NAME)
            tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
            save_variables_to_ckpt(self._model_dir)
        batch_size = 2
        feature_columns = [feature_column.numeric_column('age'), tf.feature_column.numeric_column('height')]

        def _input_fn():
            features_ds = tf.compat.v1.data.Dataset.from_tensor_slices({'age': np.array([20, 40]), 'height': np.array([4, 8])})
            labels_ds = tf.compat.v1.data.Dataset.from_tensor_slices(np.array([[213.0], [421.0]]))
            return tf.compat.v1.data.Dataset.zip((features_ds, labels_ds)).batch(batch_size).repeat(None)
        est = self._linear_regressor_fn(feature_columns=feature_columns, model_dir=self._model_dir)
        eval_metrics = est.evaluate(input_fn=_input_fn, steps=1)
        self.assertItemsEqual((metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN, metric_keys.MetricKeys.PREDICTION_MEAN, metric_keys.MetricKeys.LABEL_MEAN, tf.compat.v1.GraphKeys.GLOBAL_STEP), eval_metrics.keys())
        self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])
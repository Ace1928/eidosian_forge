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
class BaseDNNRegressorTrainTest(object):

    def __init__(self, dnn_regressor_fn, fc_impl=feature_column):
        self._dnn_regressor_fn = dnn_regressor_fn
        self._fc_impl = fc_impl

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def test_from_scratch_with_default_optimizer(self):
        hidden_units = (2, 2)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=hidden_units, feature_columns=(self._fc_impl.numeric_column('age'),), model_dir=self._model_dir)
        num_steps = 5
        dnn_regressor.train(input_fn=lambda: ({'age': ((1,),)}, ((10,),)), steps=num_steps)
        _assert_checkpoint(self, num_steps, input_units=1, hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)

    def test_from_scratch(self):
        hidden_units = (2, 2)
        opt = mock_optimizer(self, hidden_units=hidden_units)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=hidden_units, feature_columns=(self._fc_impl.numeric_column('age'),), optimizer=opt, model_dir=self._model_dir)
        self.assertEqual(0, opt.minimize.call_count)
        num_steps = 5
        summary_hook = _SummaryHook()
        dnn_regressor.train(input_fn=lambda: ({'age': ((1,),)}, ((5.0,),)), steps=num_steps, hooks=(summary_hook,))
        self.assertEqual(1, opt.minimize.call_count)
        _assert_checkpoint(self, num_steps, input_units=1, hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)
        summaries = summary_hook.summaries()
        self.assertEqual(num_steps, len(summaries))
        for summary in summaries:
            summary_keys = [v.tag for v in summary.value]
            self.assertIn(metric_keys.MetricKeys.LOSS, summary_keys)
            self.assertIn(metric_keys.MetricKeys.LOSS_MEAN, summary_keys)

    def test_one_dim(self):
        """Asserts train loss for one-dimensional input and logits."""
        base_global_step = 100
        hidden_units = (2, 2)
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), base_global_step, self._model_dir)
        expected_loss = 9.4864
        opt = mock_optimizer(self, hidden_units=hidden_units, expected_loss=expected_loss)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=hidden_units, feature_columns=(self._fc_impl.numeric_column('age'),), optimizer=opt, model_dir=self._model_dir)
        self.assertEqual(0, opt.minimize.call_count)
        num_steps = 5
        summary_hook = _SummaryHook()
        dnn_regressor.train(input_fn=lambda: ({'age': [[10.0]]}, [[1.0]]), steps=num_steps, hooks=(summary_hook,))
        self.assertEqual(1, opt.minimize.call_count)
        summaries = summary_hook.summaries()
        self.assertEqual(num_steps, len(summaries))
        for summary in summaries:
            _assert_simple_summary(self, {metric_keys.MetricKeys.LOSS_MEAN: expected_loss, 'dnn/dnn/hiddenlayer_0/fraction_of_zero_values': 0.0, 'dnn/dnn/hiddenlayer_1/fraction_of_zero_values': 0.5, 'dnn/dnn/logits/fraction_of_zero_values': 0.0, metric_keys.MetricKeys.LOSS: expected_loss}, summary)
        _assert_checkpoint(self, base_global_step + num_steps, input_units=1, hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)

    def test_multi_dim(self):
        """Asserts train loss for multi-dimensional input and logits."""
        base_global_step = 100
        hidden_units = (2, 2)
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        input_dimension = 2
        label_dimension = 3
        expected_loss = 4.3929
        opt = mock_optimizer(self, hidden_units=hidden_units, expected_loss=expected_loss)
        dnn_regressor = self._dnn_regressor_fn(hidden_units=hidden_units, feature_columns=[self._fc_impl.numeric_column('age', shape=[input_dimension])], label_dimension=label_dimension, optimizer=opt, model_dir=self._model_dir)
        self.assertEqual(0, opt.minimize.call_count)
        num_steps = 5
        summary_hook = _SummaryHook()
        dnn_regressor.train(input_fn=lambda: ({'age': [[10.0, 8.0]]}, [[1.0, -1.0, 0.5]]), steps=num_steps, hooks=(summary_hook,))
        self.assertEqual(1, opt.minimize.call_count)
        summaries = summary_hook.summaries()
        self.assertEqual(num_steps, len(summaries))
        for summary in summaries:
            _assert_simple_summary(self, {metric_keys.MetricKeys.LOSS_MEAN: expected_loss / label_dimension, 'dnn/dnn/hiddenlayer_0/fraction_of_zero_values': 0.0, 'dnn/dnn/hiddenlayer_1/fraction_of_zero_values': 0.5, 'dnn/dnn/logits/fraction_of_zero_values': 0.0, metric_keys.MetricKeys.LOSS: expected_loss}, summary)
        _assert_checkpoint(self, base_global_step + num_steps, input_units=input_dimension, hidden_units=hidden_units, output_units=label_dimension, model_dir=self._model_dir)
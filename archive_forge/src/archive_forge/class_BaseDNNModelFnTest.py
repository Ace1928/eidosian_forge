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
class BaseDNNModelFnTest(object):
    """Tests that _dnn_model_fn passes expected logits to mock head."""

    def __init__(self, dnn_model_fn, fc_impl=feature_column):
        self._dnn_model_fn = dnn_model_fn
        self._fc_impl = fc_impl

    def setUp(self):
        self._model_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self._model_dir:
            tf.compat.v1.summary.FileWriterCache.clear()
            shutil.rmtree(self._model_dir)

    def _test_logits(self, mode, hidden_units, logits_dimension, inputs, expected_logits):
        """Tests that the expected logits are passed to mock head."""
        with tf.Graph().as_default():
            tf.compat.v1.train.create_global_step()
            head = mock_head(self, hidden_units=hidden_units, logits_dimension=logits_dimension, expected_logits=expected_logits)
            estimator_spec = self._dnn_model_fn(features={'age': tf.constant(inputs)}, labels=tf.constant([[1]]), mode=mode, head=head, hidden_units=hidden_units, feature_columns=[self._fc_impl.numeric_column('age', shape=np.array(inputs).shape[1:])], optimizer=mock_optimizer(self, hidden_units))
            with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=self._model_dir) as sess:
                if mode == ModeKeys.TRAIN:
                    sess.run(estimator_spec.train_op)
                elif mode == ModeKeys.EVAL:
                    sess.run(estimator_spec.loss)
                elif mode == ModeKeys.PREDICT:
                    sess.run(estimator_spec.predictions)
                else:
                    self.fail('Invalid mode: {}'.format(mode))

    def test_one_dim_logits(self):
        """Tests one-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)]] = [[6.1, 4.9]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)]]
                   = [[relu(2.38), relu(-0.12)]] = [[2.38, 0]]
    logits = [[-1*2.38 +1*0 +0.3]] = [[-2.08]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=1, inputs=[[10.0]], expected_logits=[[-2.08]])

    def test_multi_dim_logits(self):
        """Tests multi-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)]] = [[6.1, 4.9]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)]]
                   = [[relu(2.38), relu(-0.12)]] = [[2.38, 0]]
    logits = [[-1*2.38 +0.3, 1*2.38 -0.3, 0.5*2.38]]
           = [[-2.08, 2.08, 1.19]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=3, inputs=[[10.0]], expected_logits=[[-2.08, 2.08, 1.19]])

    def test_multi_example_multi_dim_logits(self):
        """Tests multiple examples and multi-dimensional logits.

    input_layer = [[10], [5]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)],
                      [relu(0.6*5 +0.1), relu(0.5*5 -0.1)]]
                   = [[6.1, 4.9], [3.1, 2.4]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)],
                      [relu(1*3.1 -0.8*2.4 +0.2), relu(0.8*3.1 -1*2.4 -0.1)]]
                   = [[2.38, 0], [1.38, 0]]
    logits = [[-1*2.38 +0.3, 1*2.38 -0.3, 0.5*2.38],
              [-1*1.38 +0.3, 1*1.38 -0.3, 0.5*1.38]]
           = [[-2.08, 2.08, 1.19], [-1.08, 1.08, 0.69]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=3, inputs=[[10.0], [5.0]], expected_logits=[[-2.08, 2.08, 1.19], [-1.08, 1.08, 0.69]])

    def test_multi_dim_input_one_dim_logits(self):
        """Tests multi-dimensional inputs and one-dimensional logits.

    input_layer = [[10, 8]]
    hidden_layer_0 = [[relu(0.6*10 -0.6*8 +0.1), relu(0.5*10 -0.5*8 -0.1)]]
                   = [[1.3, 0.9]]
    hidden_layer_1 = [[relu(1*1.3 -0.8*0.9 + 0.2), relu(0.8*1.3 -1*0.9 -0.2)]]
                   = [[0.78, relu(-0.06)]] = [[0.78, 0]]
    logits = [[-1*0.78 +1*0 +0.3]] = [[-0.48]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=1, inputs=[[10.0, 8.0]], expected_logits=[[-0.48]])

    def test_multi_dim_input_multi_dim_logits(self):
        """Tests multi-dimensional inputs and multi-dimensional logits.

    input_layer = [[10, 8]]
    hidden_layer_0 = [[relu(0.6*10 -0.6*8 +0.1), relu(0.5*10 -0.5*8 -0.1)]]
                   = [[1.3, 0.9]]
    hidden_layer_1 = [[relu(1*1.3 -0.8*0.9 + 0.2), relu(0.8*1.3 -1*0.9 -0.2)]]
                   = [[0.78, relu(-0.06)]] = [[0.78, 0]]
    logits = [[-1*0.78 + 0.3, 1*0.78 -0.3, 0.5*0.78]] = [[-0.48, 0.48, 0.39]]
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            self._test_logits(mode, hidden_units=(2, 2), logits_dimension=3, inputs=[[10.0, 8.0]], expected_logits=[[-0.48, 0.48, 0.39]])

    def test_multi_feature_column_multi_dim_logits(self):
        """Tests multiple feature columns and multi-dimensional logits.

    All numbers are the same as test_multi_dim_input_multi_dim_logits. The only
    difference is that the input consists of two 1D feature columns, instead of
    one 2D feature column.
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        hidden_units = (2, 2)
        logits_dimension = 3
        inputs = ([[10.0]], [[8.0]])
        expected_logits = [[-0.48, 0.48, 0.39]]
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            with tf.Graph().as_default():
                tf.compat.v1.train.create_global_step()
                head = mock_head(self, hidden_units=hidden_units, logits_dimension=logits_dimension, expected_logits=expected_logits)
                estimator_spec = self._dnn_model_fn(features={'age': tf.constant(inputs[0]), 'height': tf.constant(inputs[1])}, labels=tf.constant([[1]]), mode=mode, head=head, hidden_units=hidden_units, feature_columns=[self._fc_impl.numeric_column('age'), self._fc_impl.numeric_column('height')], optimizer=mock_optimizer(self, hidden_units))
                with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=self._model_dir) as sess:
                    if mode == ModeKeys.TRAIN:
                        sess.run(estimator_spec.train_op)
                    elif mode == ModeKeys.EVAL:
                        sess.run(estimator_spec.loss)
                    elif mode == ModeKeys.PREDICT:
                        sess.run(estimator_spec.predictions)
                    else:
                        self.fail('Invalid mode: {}'.format(mode))

    def test_multi_feature_column_mix_multi_dim_logits(self):
        """Tests multiple feature columns and multi-dimensional logits.

    All numbers are the same as test_multi_dim_input_multi_dim_logits. The only
    difference is that the input consists of two 1D feature columns, instead of
    one 2D feature column.
    """
        base_global_step = 100
        create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
        hidden_units = (2, 2)
        logits_dimension = 3
        inputs = ([[10.0]], [[8.0]])
        expected_logits = [[-0.48, 0.48, 0.39]]
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            with tf.Graph().as_default():
                tf.compat.v1.train.create_global_step()
                head = mock_head(self, hidden_units=hidden_units, logits_dimension=logits_dimension, expected_logits=expected_logits)
                estimator_spec = self._dnn_model_fn(features={'age': tf.constant(inputs[0]), 'height': tf.constant(inputs[1])}, labels=tf.constant([[1]]), mode=mode, head=head, hidden_units=hidden_units, feature_columns=[feature_column.numeric_column('age'), tf.feature_column.numeric_column('height')], optimizer=mock_optimizer(self, hidden_units))
                with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=self._model_dir) as sess:
                    if mode == ModeKeys.TRAIN:
                        sess.run(estimator_spec.train_op)
                    elif mode == ModeKeys.EVAL:
                        sess.run(estimator_spec.loss)
                    elif mode == ModeKeys.PREDICT:
                        sess.run(estimator_spec.predictions)
                    else:
                        self.fail('Invalid mode: {}'.format(mode))

    def test_features_tensor_raises_value_error(self):
        """Tests that passing a Tensor for features raises a ValueError."""
        hidden_units = (2, 2)
        logits_dimension = 3
        inputs = ([[10.0]], [[8.0]])
        expected_logits = [[0, 0, 0]]
        with tf.Graph().as_default():
            tf.compat.v1.train.create_global_step()
            head = mock_head(self, hidden_units=hidden_units, logits_dimension=logits_dimension, expected_logits=expected_logits)
            with self.assertRaisesRegexp(ValueError, 'features should be a dict'):
                self._dnn_model_fn(features=tf.constant(inputs), labels=tf.constant([[1]]), mode=ModeKeys.TRAIN, head=head, hidden_units=hidden_units, feature_columns=[self._fc_impl.numeric_column('age', shape=np.array(inputs).shape[1:])], optimizer=mock_optimizer(self, hidden_units))
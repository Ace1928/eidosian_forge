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
def _test_complete_flow(self, n_classes, train_input_fn, eval_input_fn, predict_input_fn, input_dimension, prediction_length):
    feature_columns = [self._fc_lib.numeric_column('x', shape=(input_dimension,))]
    est = self._linear_classifier_fn(feature_columns=feature_columns, n_classes=n_classes, model_dir=self._model_dir)
    est.train(train_input_fn, steps=200)
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(200, scores[tf.compat.v1.GraphKeys.GLOBAL_STEP])
    self.assertIn(metric_keys.MetricKeys.LOSS, six.iterkeys(scores))
    predictions = np.array([x['classes'] for x in est.predict(predict_input_fn)])
    self.assertAllEqual((prediction_length, 1), predictions.shape)
    feature_spec = tf.compat.v1.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(feature_spec)
    export_dir = est.export_saved_model(tempfile.mkdtemp(), serving_input_receiver_fn)
    self.assertTrue(tf.compat.v1.gfile.Exists(export_dir))
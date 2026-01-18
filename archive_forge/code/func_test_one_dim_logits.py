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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
Creates non-trainable variables representing input statistics.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
Returns the wrapped static head.
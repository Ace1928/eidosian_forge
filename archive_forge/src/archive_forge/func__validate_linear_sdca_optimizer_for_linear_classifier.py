from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _validate_linear_sdca_optimizer_for_linear_classifier(feature_columns, n_classes, optimizer, sparse_combiner):
    """Helper function for the initialization of LinearClassifier."""
    if isinstance(optimizer, LinearSDCA):
        if sparse_combiner != 'sum':
            raise ValueError('sparse_combiner must be "sum" when optimizer is a LinearSDCA object.')
        if not feature_column_lib.is_feature_column_v2(feature_columns):
            raise ValueError('V2 feature columns required when optimizer is a LinearSDCA object.')
        if n_classes > 2:
            raise ValueError('LinearSDCA cannot be used in a multi-class setting.')
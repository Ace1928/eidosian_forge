from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _label_ids(self, labels):
    """Converts labels to integer id space."""
    if self._label_vocabulary is None:
        if not labels.dtype.is_integer:
            raise ValueError('Labels dtype should be integer. Instead got {}.'.format(labels.dtype))
        label_ids = labels
    else:
        if labels.dtype != tf.dtypes.string:
            raise ValueError('Labels dtype should be string if there is a vocabulary. Instead got {}'.format(labels.dtype))
        label_ids = lookup_ops.index_table_from_tensor(vocabulary_list=tuple(self._label_vocabulary), name='class_id_lookup').lookup(labels)
    return _assert_range(label_ids, self._n_classes)
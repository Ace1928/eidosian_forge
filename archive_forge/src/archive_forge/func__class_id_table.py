from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@property
def _class_id_table(self):
    """Creates a lookup table for class_id.

    In eager execution, this lookup table will be lazily created on the first
    call of `self._class_id_table`, and cached for later use; In graph
    execution, it will be created on demand.

    Returns:
      A hash table for lookup.
    """
    if self._cached_class_id_table is None or not tf.executing_eagerly():
        self._cached_class_id_table = lookup_ops.index_table_from_tensor(vocabulary_list=tuple(self._label_vocabulary), name='class_id_lookup')
    return self._cached_class_id_table
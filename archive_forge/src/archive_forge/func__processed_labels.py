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
def _processed_labels(self, logits, labels):
    """Converts labels to integer id space."""
    if labels is None:
        raise ValueError(base_head._LABEL_NONE_ERR_MSG)
    if isinstance(labels, tf.sparse.SparseTensor):
        label_values = labels.values
        if labels.dtype == tf.dtypes.string:
            label_ids_values = self._class_id_table.lookup(label_values)
            label_ids = tf.sparse.SparseTensor(indices=labels.indices, values=label_ids_values, dense_shape=labels.dense_shape)
            processed_labels = tf.sparse.to_indicator(label_ids, self._n_classes)
        else:
            if not label_values.dtype.is_integer:
                raise ValueError('Labels dtype should be integer. Instead got {}.'.format(label_values.dtype))
            err_msg = 'labels must be an integer SparseTensor with values in [0, {})'.format(self._n_classes)
            label_values = base_head.check_label_range(labels.values, self._n_classes, message=err_msg)
            if tf.executing_eagerly():
                processed_labels = tf.sparse.to_indicator(labels, self._n_classes)
            else:
                with tf.control_dependencies([label_values]):
                    processed_labels = tf.sparse.to_indicator(labels, self._n_classes)
        processed_labels = tf.cast(processed_labels, dtype=tf.dtypes.int64)
    else:
        err_msg = 'labels must be an integer indicator Tensor with values in [0, 1]'
        processed_labels = base_head.check_label_range(labels, 2, message=err_msg)
    return base_head.check_dense_labels_match_logits_and_reshape(labels=processed_labels, logits=logits, expected_labels_dimension=self.logits_dimension)
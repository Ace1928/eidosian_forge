from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as feature_column_v1
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export(v1=['estimator.BaselineClassifier'])
class BaselineClassifier(estimator.Estimator):
    __doc__ = BaselineClassifierV2.__doc__.replace('SUM_OVER_BATCH_SIZE', 'SUM')

    def __init__(self, model_dir=None, n_classes=2, weight_column=None, label_vocabulary=None, optimizer='Ftrl', config=None, loss_reduction=tf.compat.v1.losses.Reduction.SUM):
        head = head_lib._binary_logistic_or_multi_class_head(n_classes, weight_column, label_vocabulary, loss_reduction)

        def _model_fn(features, labels, mode, config):
            return _baseline_model_fn(features=features, labels=labels, mode=mode, head=head, optimizer=optimizer, weight_column=weight_column, config=config)
        super(BaselineClassifier, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)
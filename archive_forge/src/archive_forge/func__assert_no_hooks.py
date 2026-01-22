from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
def _assert_no_hooks(test_case, spec):
    test_case.assertAllEqual([], spec.training_chief_hooks)
    test_case.assertAllEqual([], spec.training_hooks)
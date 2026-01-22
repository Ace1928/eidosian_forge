from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
class SupervisedInputReceiver(collections.namedtuple('SupervisedInputReceiver', ['features', 'labels', 'receiver_tensors'])):
    """A return type for a training_input_receiver_fn or eval_input_receiver_fn.

  This differs from a ServingInputReceiver in that (1) this receiver expects
  a set of labels to be passed in with features, and (2) this receiver does
  not support receiver_tensors_alternatives, which are primarily used for
  serving.

  The expected return values are:
    features: A `Tensor`, `SparseTensor`, or dict of string or int to `Tensor`
      or `SparseTensor`, specifying the features to be passed to the model.
    labels: A `Tensor`, `SparseTensor`, or dict of string or int to `Tensor` or
      `SparseTensor`, specifying the labels to be passed to the model.
    receiver_tensors: A `Tensor`, `SparseTensor`, or dict of string to `Tensor`
      or `SparseTensor`, specifying input nodes where this receiver expects to
      be fed by default.  Typically, this is a single placeholder expecting
      serialized `tf.Example` protos.

  """

    def __new__(cls, features, labels, receiver_tensors):
        wrap_and_check_input_tensors(features, 'feature', allow_int_keys=True)
        wrap_and_check_input_tensors(labels, 'label', allow_int_keys=True)
        receiver_tensors = wrap_and_check_input_tensors(receiver_tensors, 'receiver_tensor')
        return super(SupervisedInputReceiver, cls).__new__(cls, features=features, labels=labels, receiver_tensors=receiver_tensors)
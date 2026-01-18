from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
See `replace`.

    N.B.: This implementation assumes that for key named "foo", the underlying
    property the RunConfig holds is "_foo" (with one leading underscore).

    Args:
      config: The RunConfig to replace the values of.
      allowed_properties_list: The property name list allowed to be replaced.
      **kwargs: keyword named properties with new values.

    Raises:
      ValueError: If any property name in `kwargs` does not exist or is not
        allowed to be replaced, or both `save_checkpoints_steps` and
        `save_checkpoints_secs` are set.

    Returns:
      a new instance of `RunConfig`.
    
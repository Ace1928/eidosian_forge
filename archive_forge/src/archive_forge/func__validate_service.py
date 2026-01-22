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
def _validate_service(service):
    """Validates the service key."""
    if service is not None and (not isinstance(service, dict)):
        raise TypeError('If "service" is set in TF_CONFIG, it must be a dict. Given %s' % type(service))
    return service
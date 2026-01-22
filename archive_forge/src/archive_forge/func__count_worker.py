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
def _count_worker(cluster_spec, chief_task_type):
    """Counts the number of workers (including chief) in cluster_spec."""
    if not cluster_spec:
        raise RuntimeError('Internal error: `_count_worker` does not expect empty cluster_spec.')
    return len(cluster_spec.as_dict().get(TaskType.WORKER, [])) + len(cluster_spec.as_dict().get(chief_task_type, []))
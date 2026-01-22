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
def _get_default_session_config_distributed(self):
    """Returns None or tf.ConfigProto instance with default device_filters set.

    Device filters are set such that chief/master and worker communicates with
    only ps. session_config=None for evaluators or any other TaskType.
    """
    rewrite_opts = rewriter_config_pb2.RewriterConfig(meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE)
    graph_opts = tf.compat.v1.GraphOptions(rewrite_options=rewrite_opts)
    device_filters = None
    if self._task_type == TaskType.MASTER:
        device_filters = ['/job:ps', '/job:master']
    elif self._task_type == TaskType.CHIEF:
        device_filters = ['/job:ps', '/job:chief']
    elif self._task_type == TaskType.WORKER:
        device_filters = ['/job:ps', '/job:worker/task:%d' % self._task_id]
    elif self._task_type == TaskType.PS:
        device_filters = ['/job:ps', '/job:worker', '/job:chief', '/job:master']
    else:
        return None
    return tf.compat.v1.ConfigProto(allow_soft_placement=True, graph_options=graph_opts, device_filters=device_filters)
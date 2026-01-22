from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
class _TPUEstimatorReplicaContext(tf.distribute.ReplicaContext):
    """Internal context for storing replica id.

  This is to set eager.context.Context() so that only summary ops from
  0th replica is executed.
  """

    def __init__(self, replica_id_in_sync):
        """Creates internal replica context for TPUEstimator.

    Args:
      replica_id_in_sync: Zero indexed integer id of replica that is running the
        TPU compuation.
    """
        super(_TPUEstimatorReplicaContext, self).__init__(None, replica_id_in_sync)
        self._thread_context = distribute_lib._DefaultReplicaThreadMode()
        self._strategy = self._thread_context.strategy

    def __enter__(self):

        def replica_id_is_zero():
            return tf.math.equal(self.replica_id_in_sync_group, tf.constant(0))
        if hasattr(summary_ops_v2, '_summary_state'):
            summary_state = summary_ops_v2._summary_state
            self._summary_recording_distribution_strategy = summary_state.is_recording_distribution_strategy
            summary_state.is_recording_distribution_strategy = replica_id_is_zero

    def __exit__(self, exception_type, exception_value, traceback):
        if hasattr(summary_ops_v2, '_summary_state'):
            summary_state = summary_ops_v2._summary_state
            summary_state.is_recording_distribution_strategy = self._summary_recording_distribution_strategy
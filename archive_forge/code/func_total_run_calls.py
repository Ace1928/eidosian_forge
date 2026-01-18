import os
import signal
import sys
import threading
import time
from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@property
@deprecated(None, 'Track steps using a tf.Variable saved in checkpoint instead.')
@doc_controls.do_not_generate_docs
def total_run_calls(self):
    """Returns the number of times `PreemptionCheckpointHandler.run` is called.

    DEPRECATED: user should track total steps themselves, as this API provides
    little expressivity gain but could easily be misused and incurs extra
    synchronization cost for TPUStrategy users.

    This value tracks the number of all calls to
    `PreemptionCheckpointHandler.run` including those before the program is
    restarted and the training is restored, by saving and reading the value in
    the checkpoint. A user can compute their total number of iterations
    by `PreemptionCheckpointHandler.total_run_calls *
    number_of_steps_in_train_function`,
    while `number_of_steps_in_train_function` should be one for
    `tf.distribute.MultiWorkerMirroredStrategy` users. They can also use this
    value to infer the starting epoch and step after training restores, as shown
    in the example above.
    """
    if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
        raise NotImplementedError('Please create variables saved in checkpoint to keep track of steps and epochs.')
    return self._run_counter
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
def _stop_cluster_wise_termination_watcher_thread(self):
    """Stop the thread that is _watch_step_to_save_key."""
    if getattr(self, '_cluster_wise_termination_watcher_thread', None):
        try:
            context.context().set_config_key_value(_INITIAL_RUN_COUNT_KEY, _STOP_WATCHING_CLUSTER_VALUE)
        except (errors.AlreadyExistsError, errors.UnavailableError):
            pass
        except Exception as e:
            logging.info('Ignoring error when shutting down _stop_cluster_wise_termination_watcher_thread: ' + str(e))
        try:
            context.context().set_config_key_value(_FINAL_RUN_COUNT_KEY, _STOP_WATCHING_CLUSTER_VALUE)
        except (errors.AlreadyExistsError, errors.UnavailableError):
            pass
        except Exception as e:
            logging.info('Ignoring error when shutting down _stop_cluster_wise_termination_watcher_thread: ' + str(e))
        finally:
            self._cluster_wise_termination_watcher_thread.join()
            self._cluster_wise_termination_watcher_thread = None
            logging.info("Shut down watcher for peer's termination signal.")
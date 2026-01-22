import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _maybe_delay(self):
    """Delay if corresponding env vars are set."""
    delay_secs = int(os.environ.get('TF_COORDINATOR_SCHEDULE_START_DELAY', '0'))
    delay_secs *= self.worker_index
    delay_cap = int(os.environ.get('TF_COORDINATOR_SCHEDULE_START_DELAY_MAX', '0'))
    if delay_cap:
        delay_secs = min(delay_secs, delay_cap)
    if delay_secs > 0:
        logging.info(' Worker %d sleeping for %d seconds before running function', self.worker_index, delay_secs)
    time.sleep(delay_secs)
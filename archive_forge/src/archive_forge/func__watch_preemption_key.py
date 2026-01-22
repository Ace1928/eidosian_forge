import threading
from absl import logging
from tensorflow.python.distribute.failure_handling.failure_handling_util import detect_platform
from tensorflow.python.distribute.failure_handling.failure_handling_util import PlatformDevice
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework.errors import AbortedError
from tensorflow.python.framework.errors import CancelledError
from tensorflow.python.framework.errors import InternalError
from tensorflow.python.framework.errors import UnavailableError
from tensorflow.python.util.tf_export import tf_export
def _watch_preemption_key(self):
    logging.info('Watching preemption signal.')
    message = context.context().get_config_key_value(_PREEMPTION_KEY)
    _preemption_handling_counter.get_cell().increase_by(1)
    logging.info('Preemption signal received.')
    self._preemption_message = message
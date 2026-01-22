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
def _is_worker_failure(error):
    """Whether the error is considered a worker failure."""
    if _handle_graph_execution_error_as_worker_failure() and isinstance(error, errors.UnknownError) and ('Graph execution error' in str(error)):
        logging.info(f'Handling {type(error)}: {str(error)} as worker failure.')
        return True
    if isinstance(error, (ClosureInputError, ClosureAbortedError)):
        error = error.original_exception
    if _JOB_WORKER_STRING_IDENTIFIER not in str(error):
        return False
    if _RPC_ERROR_FROM_PS in str(error):
        return False
    if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
        return True
    if isinstance(error, errors.InvalidArgumentError):
        if 'unknown device' in str(error).lower() or 'Primary device is not remote' in str(error) or 'Unable to find the relevant tensor remote_handle' in str(error):
            return True
    if isinstance(error, errors.NotFoundError):
        if 'is neither a type of a primitive operation nor a name of a function registered' in str(error):
            return True
    if isinstance(error, errors.CancelledError):
        return True
    if isinstance(error, TypeError) and 'Binding inputs to tf.function' in str(error):
        return True
    return False
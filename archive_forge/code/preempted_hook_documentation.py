import logging as _logging
import os
import threading
import time
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
A thread that polls the state of a TPU node.

  When the node transitions into a TERMINAL state (PREEMPTED, TERMINATED)
  that's considered as not recoverable by the underlying infrastructure,
  it attempts to close the session, and exits the entire process if the
  session.close() stucks.
  
import logging as _logging
import os
import threading
import time
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
class CloudTPUPreemptedHook(session_run_hook.SessionRunHook):
    """The SessionRunHook for preemptible Cloud TPUs.

  This is an implementation of SessionRunHook for the pre-emptible Google Cloud
  TPU service. It attempts to close the session if the TPU is preempted, and
  exits the coordinator process if the session cannot be closed.
  """

    def __init__(self, cluster):
        self._cluster = cluster

    def after_create_session(self, session, coord):
        if tpu_cluster_resolver.is_running_in_gce():
            self._tpu_poller = _TPUPollingThread(self._cluster, session)
            self._tpu_poller.start()

    def end(self, session):
        self._tpu_poller.stop()
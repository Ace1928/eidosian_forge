import os
import threading
import time
from typing import Any, List, Optional, Text
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache
def _save_fn():
    """Run the saver process."""
    logging.info('Saving checkpoints for %d into %s.', step, self._save_path)
    start_time = time.time()
    for l in self._listeners:
        l.before_save(session, step)
    self._get_saver().save(session, self._save_path, global_step=step)
    if self._summary_writer is None:
        raise ValueError('Summary writer is not initialised')
    self._summary_writer.add_session_log(event_pb2.SessionLog(status=event_pb2.SessionLog.CHECKPOINT, checkpoint_path=self._save_path), step)
    for l in self._listeners:
        l.after_save(session, step)
    end_time = time.time()
    metrics.AddAsyncCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT_V1, microseconds=_get_duration_microseconds(start_time, end_time))
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
        metrics.AddTrainingTimeSaved(api_label=_ASYNC_CHECKPOINT_V1, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE, start_time))
    _END_TIME_OF_LAST_WRITE = start_time
    logging.info('Checkpoint actual writing time: (%.3f sec)', end_time - start_time)
    logging.info('Checkpoint finished for %d into %s.', step, self._save_path)
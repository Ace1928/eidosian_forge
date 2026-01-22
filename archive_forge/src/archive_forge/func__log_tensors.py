import os
import time
import numpy as np
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export
def _log_tensors(self, tensor_values):
    original = np.get_printoptions()
    np.set_printoptions(suppress=True)
    elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    if self._formatter:
        logging.info(self._formatter(tensor_values))
    else:
        stats = []
        for tag in self._tag_order:
            stats.append('%s = %s' % (tag, tensor_values[tag]))
        if elapsed_secs is not None:
            logging.info('%s (%.3f sec)', ', '.join(stats), elapsed_secs)
        else:
            logging.info('%s', ', '.join(stats))
    np.set_printoptions(**original)
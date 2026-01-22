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
def _get_saver(self):
    if self._saver is not None:
        return self._saver
    elif self._scaffold is not None:
        return self._scaffold.saver
    collection_key = ops.GraphKeys.SAVERS
    savers = ops.get_collection(collection_key)
    if not savers:
        raise RuntimeError('No items in collection {}. Please add a saver to the collection or provide a saver or scaffold.'.format(collection_key))
    elif len(savers) > 1:
        raise RuntimeError('More than one item in collection {}. Please indicate which one to use by passing it to the constructor.'.format(collection_key))
    self._saver = savers[0]
    return savers[0]
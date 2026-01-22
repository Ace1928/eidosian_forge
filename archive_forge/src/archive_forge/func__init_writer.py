import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
def _init_writer(self, model):
    """Sets file writer."""
    if context.executing_eagerly():
        self.writer = summary_ops_v2.create_file_writer_v2(self.log_dir)
        if not model.run_eagerly and self.write_graph:
            with self.writer.as_default():
                summary_ops_v2.graph(K.get_graph())
    elif self.write_graph:
        self.writer = tf_summary.FileWriter(self.log_dir, K.get_graph())
    else:
        self.writer = tf_summary.FileWriter(self.log_dir)
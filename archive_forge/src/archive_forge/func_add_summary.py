from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
def add_summary(self, summ, current_global_step):
    """Add summary."""
    if isinstance(summ, bytes):
        summary_proto = summary_pb2.Summary()
        summary_proto.ParseFromString(summ)
        summ = summary_proto
    if current_global_step in self._summaries:
        step_summaries = self._summaries[current_global_step]
    else:
        step_summaries = []
        self._summaries[current_global_step] = step_summaries
    step_summaries.append(summ)
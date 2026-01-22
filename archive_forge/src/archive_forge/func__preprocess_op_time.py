import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _preprocess_op_time(self, op_time):
    """Update the start and end time of ops in step stats.

    Args:
    op_time: How the execution time of op is shown in timeline. Possible values
      are "schedule", "gpu" and "all". "schedule" will show op from the time it
      is scheduled to the end of the scheduling. Notice by the end of its
      scheduling its async kernels may not start yet. It is shown using the
      default value from step_stats. "gpu" will show op with the execution time
      of its kernels on GPU. "all" will show op from the start of its scheduling
      to the end of its last kernel.
    """
    if op_time == 'schedule':
        self._step_stats = self._origin_step_stats
        return
    self._step_stats = copy.deepcopy(self._origin_step_stats)
    stream_all_stats = []
    job_stats = []
    for stats in self._step_stats.dev_stats:
        if '/stream:all' in stats.device:
            stream_all_stats.append(stats)
        elif '/job' in stats.device:
            job_stats.append(stats)
    op_gpu_start = {}
    op_gpu_end = {}
    for stats in stream_all_stats:
        for kernel in stats.node_stats:
            name, _ = self._parse_kernel_label(kernel.timeline_label, kernel.node_name)
            start = kernel.all_start_micros
            end = kernel.all_start_micros + kernel.all_end_rel_micros
            if name in op_gpu_start:
                op_gpu_start[name] = min(op_gpu_start[name], start)
                op_gpu_end[name] = max(op_gpu_end[name], end)
            else:
                op_gpu_start[name] = start
                op_gpu_end[name] = end
    for stats in job_stats:
        for op in stats.node_stats:
            if op.node_name in op_gpu_start:
                end = max(op_gpu_end[op.node_name], op.all_start_micros + op.all_end_rel_micros)
                if op_time == 'gpu':
                    op.all_start_micros = op_gpu_start[op.node_name]
                op.all_end_rel_micros = end - op.all_start_micros
import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _emit_op(self, nodestats, pid, is_gputrace):
    """Generates a Chrome Trace event to show Op execution.

    Args:
      nodestats: The 'NodeExecStats' proto recording op execution.
      pid: The pid assigned for the device where this op ran.
      is_gputrace: If True then this op came from the GPUTracer.
    """
    node_name = nodestats.node_name
    start = nodestats.all_start_micros
    duration = nodestats.all_end_rel_micros
    tid = nodestats.thread_id
    inputs = []
    if is_gputrace:
        node_name, op = self._parse_kernel_label(nodestats.timeline_label, node_name)
    elif node_name == 'RecvTensor':
        op = 'RecvTensor'
    else:
        _, op, inputs = self._parse_op_label(nodestats.timeline_label)
    args = {'name': node_name, 'op': op}
    if build_info.build_info['is_rocm_build']:
        args['kernel'] = nodestats.timeline_label.split('@@')[0]
    for i, iname in enumerate(inputs):
        args['input%d' % i] = iname
    self._chrome_trace.emit_region(start, duration, pid, tid, 'Op', op, args)
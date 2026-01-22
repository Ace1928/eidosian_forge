import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def _write_tensor_list_section(self, graph_order):
    """Writes the tensor-list section of the report."""
    self._write_report('%s %s\n' % (_MARKER_SECTION_BEGIN, _SECTION_NAME_TENSOR_LIST))
    self._write_report('%s %d\n' % (_FIELD_NAME_NUM_TENSORS, len(graph_order.tensors)))
    for i in range(0, len(graph_order.tensors)):
        tensor = graph_order.tensors[i]
        line = '%d "%s"' % (i, tensor.name)
        consumers = tensor.consumers()
        consumers.sort(key=lambda op: op.name)
        for consumer_op in consumers:
            if consumer_op.name not in graph_order.op_to_idx:
                raise ValueError('consumer_op is not in op_to_idx.  got consumer_op={}, op_to_idx={}'.format(consumer_op.name, graph_order.op_to_idx))
            line += ' %d' % graph_order.op_to_idx[consumer_op.name]
        line += '\n'
        self._write_report(line)
    self._write_report('%s %s\n' % (_MARKER_SECTION_END, _SECTION_NAME_TENSOR_LIST))
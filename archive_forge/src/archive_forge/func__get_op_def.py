import re
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
def _get_op_def(op):
    return op.op_def or op_def_registry.get(op.type)
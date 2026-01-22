import re
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
def _make_argname_from_tensor_name(name):
    return re.sub(':0$', '', name).replace(':', '_o')
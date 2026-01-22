import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _PopulateTFImportGraphDefOptions(options, prefix, input_map, return_elements, validate_colocation_constraints, propagate_device_spec=False):
    """Populates the TF_ImportGraphDefOptions `options`."""
    c_api.TF_ImportGraphDefOptionsSetPrefix(options, prefix)
    c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options, True)
    c_api.TF_ImportGraphDefOptionsSetPropagateDeviceSpec(options, propagate_device_spec)
    for input_src, input_dst in input_map.items():
        input_src = compat.as_str(input_src)
        if input_src.startswith('^'):
            src_name = compat.as_str(input_src[1:])
            dst_op = input_dst._as_tf_output().oper
            c_api.TF_ImportGraphDefOptionsRemapControlDependency(options, src_name, dst_op)
        else:
            src_name, src_idx = _ParseTensorName(input_src)
            src_name = compat.as_str(src_name)
            dst_output = input_dst._as_tf_output()
            c_api.TF_ImportGraphDefOptionsAddInputMapping(options, src_name, src_idx, dst_output)
    for name in return_elements or []:
        if ':' in name:
            op_name, index = _ParseTensorName(name)
            op_name = compat.as_str(op_name)
            c_api.TF_ImportGraphDefOptionsAddReturnOutput(options, op_name, index)
        else:
            c_api.TF_ImportGraphDefOptionsAddReturnOperation(options, compat.as_str(name))
    c_api.TF_ImportGraphDefOptionsSetValidateColocationConstraints(options, validate_colocation_constraints)
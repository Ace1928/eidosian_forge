import re
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
def _tensor_to_argdef(t, name=None, used_names=None):
    """Convert tensor t to an argdef, with a specified name or a unique name."""
    arg = op_def_pb2.OpDef.ArgDef()
    if name is None:
        arg.name = _make_argname_from_tensor_name(t.name)
        if used_names is not None:
            if arg.name in used_names:
                i = 0
                while True:
                    new_name = '%s_U%d' % (arg.name, i)
                    if new_name not in used_names:
                        arg.name = new_name
                        break
                    i += 1
            used_names.add(arg.name)
    else:
        arg.name = name
    arg.type = t.dtype.as_datatype_enum
    return arg
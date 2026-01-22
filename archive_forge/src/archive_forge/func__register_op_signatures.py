import inspect
from . import _numpy_op_doc
from . import numpy as mx_np
from . import numpy_extension as mx_npx
from .base import _NP_OP_SUBMODULE_LIST, _NP_EXT_OP_SUBMODULE_LIST, _get_op_submodule_name
def _register_op_signatures():
    for op_name in dir(_numpy_op_doc):
        op = _get_builtin_op(op_name)
        if op is not None:
            op.__signature__ = inspect.signature(getattr(_numpy_op_doc, op_name))
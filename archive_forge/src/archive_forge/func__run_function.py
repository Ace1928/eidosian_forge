import collections
import copyreg
import io
import pickle
import sys
import threading
import traceback
from enum import Enum
import torch
import torch.distributed as dist
from torch._C._distributed_rpc import _get_current_rpc_agent
def _run_function(python_udf):
    """
    This function is exclusively called from C++.
    See ``torch/csrc/distributed/rpc/python_rpc_handler.cpp``.

    Runs a Python UDF and returns its return value.
    Wraps any exception in ``RemoteException`` if the function raises.
    """
    try:
        if isinstance(python_udf, AttributeError):
            raise python_udf
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        except_str = f'On {_get_current_rpc_agent().get_worker_info()}:\n{repr(e)}\n{traceback.format_exc()}'
        print(except_str, file=sys.stderr)
        result = RemoteException(except_str, type(e))
    return result
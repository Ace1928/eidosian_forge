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
def _script_module_reducer(self, script_module):
    """
        Serializes a ScriptModule.
        """
    f = io.BytesIO()
    torch.jit.save(script_module, f)
    return (_InternalRPCPickler._script_module_receiver, (f.getvalue(),))
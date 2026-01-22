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
def _tensor_reducer(self, tensor):
    global _thread_local_tensor_tables
    _thread_local_tensor_tables.send_tables.append(tensor)
    tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
    return (_InternalRPCPickler._tensor_receiver, (tensor_index,))
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
class RPCExecMode(Enum):
    SYNC = 'sync'
    ASYNC = 'async'
    ASYNC_JIT = 'async_jit'
    REMOTE = 'remote'
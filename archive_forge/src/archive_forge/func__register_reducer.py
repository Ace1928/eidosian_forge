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
def _register_reducer(self, obj_class, reducer):
    if obj_class not in self._class_reducer_dict:
        self._class_reducer_dict[obj_class] = reducer
import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
def _convert_node_to_placeholder(graph, node, inps):
    if node.op == 'output' or node.op == 'placeholder':
        return False
    if is_load_tensor_node(node):
        return False
    concrete_val = node.meta.get('concrete_value', None)
    if isinstance(concrete_val, torch.Tensor):
        node.op = 'placeholder'
        node.target = node.name
        node.args = ()
        node.kwargs = {}
        inps.append(concrete_val)
        return True
    elif concrete_val is None:
        return False
    elif concrete_val is is_tuple:
        r = False
        for tuple_user in list(node.users):
            r = _convert_node_to_placeholder(graph, tuple_user, inps) or r
        return r
    elif isinstance(concrete_val, LoadTensorMeta):
        node.op = 'call_function'
        node.target = torch.ops.debugprims.load_tensor.default
        node.args = (os.path.join('eager', node.name), concrete_val.size, concrete_val.stride)
        node.kwargs = {'device': concrete_val.device, 'dtype': concrete_val.dtype}
        return True
    return False
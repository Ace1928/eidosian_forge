import concurrent.futures
import json
import multiprocessing.connection
from typing import Any, List, Optional, Union
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
def _serialize_cuda_tensor(tensor: torch.Tensor):
    assert tensor.device.type == 'cuda'
    func, args = torch.multiprocessing.reductions.reduce_tensor(tensor)
    assert func is torch.multiprocessing.reductions.rebuild_cuda_tensor
    assert args[6] == tensor.device.index
    return args
from typing import Any, Callable, List, no_type_check
import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass
def wait_for_optim_stream_callback():
    torch.cuda.current_stream().wait_stream(optim_stream_state.optim_stream)
    for param in ddp_inst._get_data_parallel_params(ddp_inst.module):
        if hasattr(param, '_in_backward_optimizers'):
            param.grad = None
    optim_stream_state.wait_for_optim_stream_enqueued = False
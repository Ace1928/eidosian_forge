import torch
import torch.distributed as dist
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Any, no_type_check
from torch.distributed.utils import _free_storage
def wait_for_stream_cb():
    torch.cuda.current_stream().wait_stream(stream)
    for n, p in ddp_weakref().module.named_parameters():
        if hasattr(p, '_ddp_mp_hook_state'):
            p._ddp_mp_hook_state[1].remove()
            delattr(p, '_ddp_mp_hook_state')
        if not p.requires_grad and (not hasattr(p, '_ddp_ignored')):
            p.data = p._fp_param
    hook_state.wait_for_stream_enqueued = False
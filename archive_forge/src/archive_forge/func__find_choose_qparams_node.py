import logging
from typing import Optional
import torch
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import QuantizationSpecBase
from torch.fx.passes.infra.pass_base import PassResult
def _find_choose_qparams_node(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    from collections import deque
    queue = deque(list(node.users.keys()))
    while len(queue):
        n = queue.popleft()
        if n.op == 'output':
            continue
        if n.op == 'call_function' and n.target == torch.ops.quantized_decomposed.choose_qparams.tensor:
            return n
        for k in n.users.keys():
            queue.append(k)
    return None
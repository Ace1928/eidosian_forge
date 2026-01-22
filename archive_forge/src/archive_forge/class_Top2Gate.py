from typing import Callable, Dict, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """
    wg: torch.nn.Linear

    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.wg(input)
        return top2gating(logits)
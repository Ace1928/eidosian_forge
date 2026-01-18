from typing import Callable, Dict, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
def top2gating(logits: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1, dtype=torch.float)
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts)
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float('-inf'))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_classes=num_experts)
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    gates1 = gates1_s.unsqueeze(-1) * mask1
    gates2 = gates2_s.unsqueeze(-1) * mask2
    locations1_sc = one_hot(locations1_s, num_classes=capacity)
    locations2_sc = one_hot(locations2_s, num_classes=capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    return (l_aux.to(logits.dtype), combine_weights.to(logits.dtype), dispatch_mask)
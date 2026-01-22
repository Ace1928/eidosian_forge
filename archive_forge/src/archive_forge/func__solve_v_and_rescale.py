import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from ..modules import Module
def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
    v = torch.linalg.multi_dot([weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]).squeeze(1)
    return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))
from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class BaselineSoftmax(nn.Module):
    """Baseline softmax that does an output linear projection and a softmax.


        We also support LMCL (Large Margin Cosine Loss) from the CosFace paper. See
        more detailed comment in the MEVO class below.

        This is intended to be used with an embedding layer with shared weights.

    Args:
        proj_weight (nn.Parameter):
            The shared weight.
        tile_factor (int):
            Unused. It is here to make kernel init easier with MEVO.
        log_softmax (bool):
            If True, use log_softmax instead of softmax.
        margin (float):
            Used in LMCL (when scale != None). See MEVO comments for
            more details.
        scale (Optional[float]):
            Used in LMCL. If scale is None, LMCL is turned off. See
            MEVO comments for more details.

    """

    def __init__(self, proj_weight: nn.Parameter, tile_factor: int=0, log_softmax: bool=True, margin: float=0.35, scale: Optional[float]=None):
        super().__init__()
        out_dim, in_dim = proj_weight.shape
        assert 'cuda' in str(proj_weight.device), 'weight should be on GPU'
        self.fc = nn.Linear(in_dim, out_dim, bias=False).to('cuda')
        assert proj_weight.dtype in [torch.float16, torch.float32]
        if proj_weight.dtype == torch.float16:
            self.fc = self.fc.half()
        self.fc.weight = proj_weight
        assert self.fc.weight.dtype in [torch.float16, torch.float32], self.fc.weight.dtype
        self.fp16 = self.fc.weight.dtype == torch.float16
        self.log_softmax = log_softmax
        self.margin = margin
        self.scale = scale

    def lmcl_pre_softmax(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = F.normalize(input, dim=1)
        w = F.normalize(self.fc.weight, dim=1)
        logits = torch.einsum('nc,kc->nk', x, w)
        row_ind = torch.arange(x.shape[0], dtype=torch.long).to(x.device)
        col_ind = target
        logits[row_ind, col_ind] -= self.margin
        logits *= self.scale
        return logits

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function that computes softmax output with the input and target."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        if self.fp16:
            assert input.dtype == torch.float16
        if self.scale is not None:
            x = self.lmcl_pre_softmax(input, target)
        else:
            x = self.fc(input)
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1, dtype=torch.float32)
        else:
            x = F.softmax(x, dim=-1, dtype=torch.float32)
        assert x.dtype == torch.float32
        return x
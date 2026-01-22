import math
import torch
import torch.jit
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
@torch.jit.script_if_tracing
def _rejection_sample(loc, concentration, proposal_r, x):
    done = torch.zeros(x.shape, dtype=torch.bool, device=loc.device)
    while not done.all():
        u = torch.rand((3,) + x.shape, dtype=loc.dtype, device=loc.device)
        u1, u2, u3 = u.unbind()
        z = torch.cos(math.pi * u1)
        f = (1 + proposal_r * z) / (proposal_r + z)
        c = concentration * (proposal_r - f)
        accept = (c * (2 - c) - u2 > 0) | ((c / u2).log() + 1 - c >= 0)
        if accept.any():
            x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
            done = done | accept
    return (x + math.pi + loc) % (2 * math.pi) - math.pi
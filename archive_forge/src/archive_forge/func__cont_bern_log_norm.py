import math
from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
from torch.nn.functional import binary_cross_entropy_with_logits
def _cont_bern_log_norm(self):
    """computes the log normalizing constant as a function of the 'probs' parameter"""
    cut_probs = self._cut_probs()
    cut_probs_below_half = torch.where(torch.le(cut_probs, 0.5), cut_probs, torch.zeros_like(cut_probs))
    cut_probs_above_half = torch.where(torch.ge(cut_probs, 0.5), cut_probs, torch.ones_like(cut_probs))
    log_norm = torch.log(torch.abs(torch.log1p(-cut_probs) - torch.log(cut_probs))) - torch.where(torch.le(cut_probs, 0.5), torch.log1p(-2.0 * cut_probs_below_half), torch.log(2.0 * cut_probs_above_half - 1.0))
    x = torch.pow(self.probs - 0.5, 2)
    taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
    return torch.where(self._outside_unstable_region(), log_norm, taylor)
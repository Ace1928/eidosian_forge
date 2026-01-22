import math
from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
from torch.nn.functional import binary_cross_entropy_with_logits
computes the log normalizing constant as a function of the natural parameter
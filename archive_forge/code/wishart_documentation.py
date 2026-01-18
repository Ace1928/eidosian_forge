import math
import warnings
from numbers import Number
from typing import Optional, Union
import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.distributions.utils import lazy_property

        .. warning::
            In some cases, sampling algorithm based on Bartlett decomposition may return singular matrix samples.
            Several tries to correct singular samples are performed by default, but it may end up returning
            singular matrix samples. Singular samples may return `-inf` values in `.log_prob()`.
            In those cases, the user should validate the samples and either fix the value of `df`
            or adjust `max_try_correction` value for argument in `.rsample` accordingly.
        
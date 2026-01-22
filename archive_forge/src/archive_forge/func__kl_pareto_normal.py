import math
import warnings
from functools import total_ordering
from typing import Callable, Dict, Tuple, Type
import torch
from torch import inf
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_normal import HalfNormal
from .independent import Independent
from .laplace import Laplace
from .lowrank_multivariate_normal import (
from .multivariate_normal import _batch_mahalanobis, MultivariateNormal
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform
from .utils import _sum_rightmost, euler_constant as _euler_gamma
@register_kl(Pareto, Normal)
def _kl_pareto_normal(p, q):
    var_normal = 2 * q.scale.pow(2)
    common_term = p.scale / (p.alpha - 1)
    t1 = (math.sqrt(2 * math.pi) * q.scale * p.alpha / p.scale).log()
    t2 = p.alpha.reciprocal()
    t3 = p.alpha * common_term.pow(2) / (p.alpha - 2)
    t4 = (p.alpha * common_term - q.loc).pow(2)
    result = t1 - t2 + (t3 + t4) / var_normal - 1
    result[p.alpha <= 2] = inf
    return result
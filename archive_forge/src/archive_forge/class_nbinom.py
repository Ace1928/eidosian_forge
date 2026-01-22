import numpy as np
import scipy.stats
import warnings
class nbinom(NegativeBinomial):
    """
    The negative binomial link function.

    .. deprecated: 0.14.0

       Use NegativeBinomial instead.

    Notes
    -----
    g(p) = log(p/(p + 1/alpha))

    nbinom is an alias of NegativeBinomial.
    nbinom = NegativeBinomial(alpha=1.)
    """

    def __init__(self, alpha=1.0):
        _link_deprecation_warning('nbinom', 'NegativeBinomial')
        super().__init__(alpha=alpha)
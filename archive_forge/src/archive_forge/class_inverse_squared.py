import numpy as np
import scipy.stats
import warnings
class inverse_squared(InverseSquared):
    """
    Deprecated alias of InverseSquared.

    .. deprecated: 0.14.0

        Use InverseSquared instead.
    """

    def __init__(self):
        _link_deprecation_warning('inverse_squared', 'InverseSquared')
        super().__init__()
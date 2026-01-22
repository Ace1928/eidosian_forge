import numpy as np
import scipy.stats
import warnings
class cloglog(CLogLog):
    """
    The CLogLog transform link function.

    .. deprecated: 0.14.0

       Use CLogLog instead.

    Notes
    -----
    g(`p`) = log(-log(1-`p`))

    cloglog is an alias for CLogLog
    cloglog = CLogLog()
    """

    def __init__(self):
        _link_deprecation_warning('cloglog', 'CLogLog')
        super().__init__()
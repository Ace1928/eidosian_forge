import numpy as np
from ._penalties import NonePenalty
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime
minimize negative penalized log-likelihood

        Parameters
        ----------
        method : None or str
            Method specifies the scipy optimizer as in nonlinear MLE models.
        trim : {bool, float}
            Default is False or None, which uses no trimming.
            If trim is True or a float, then small parameters are set to zero.
            If True, then a default threshold is used. If trim is a float, then
            it will be used as threshold.
            The default threshold is currently 1e-4, but it will change in
            future and become penalty function dependent.
        kwds : extra keyword arguments
            This keyword arguments are treated in the same way as in the
            fit method of the underlying model class.
            Specifically, additional optimizer keywords and cov_type related
            keywords can be added.
        
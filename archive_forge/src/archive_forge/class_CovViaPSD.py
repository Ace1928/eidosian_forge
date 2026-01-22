from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
class CovViaPSD(Covariance):
    """
    Representation of a covariance provided via an instance of _PSD
    """

    def __init__(self, psd):
        self._LP = psd.U
        self._log_pdet = psd.log_pdet
        self._rank = psd.rank
        self._covariance = psd._M
        self._shape = psd._M.shape
        self._psd = psd
        self._allow_singular = False

    def _whiten(self, x):
        return x @ self._LP

    def _support_mask(self, x):
        return self._psd._support_mask(x)
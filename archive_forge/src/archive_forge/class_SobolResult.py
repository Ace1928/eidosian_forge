from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
import numpy as np
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
@dataclass
class SobolResult:
    first_order: np.ndarray
    total_order: np.ndarray
    _indices_method: Callable
    _f_A: np.ndarray
    _f_B: np.ndarray
    _f_AB: np.ndarray
    _A: np.ndarray | None = None
    _B: np.ndarray | None = None
    _AB: np.ndarray | None = None
    _bootstrap_result: BootstrapResult | None = None

    def bootstrap(self, confidence_level: DecimalNumber=0.95, n_resamples: IntNumber=999) -> BootstrapSobolResult:
        """Bootstrap Sobol' indices to provide confidence intervals.

        Parameters
        ----------
        confidence_level : float, default: ``0.95``
            The confidence level of the confidence intervals.
        n_resamples : int, default: ``999``
            The number of resamples performed to form the bootstrap
            distribution of the indices.

        Returns
        -------
        res : BootstrapSobolResult
            Bootstrap result containing the confidence intervals and the
            bootstrap distribution of the indices.

            An object with attributes:

            first_order : BootstrapResult
                Bootstrap result of the first order indices.
            total_order : BootstrapResult
                Bootstrap result of the total order indices.
            See `BootstrapResult` for more details.

        """

        def statistic(idx):
            f_A_ = self._f_A[:, idx]
            f_B_ = self._f_B[:, idx]
            f_AB_ = self._f_AB[..., idx]
            return self._indices_method(f_A_, f_B_, f_AB_)
        n = self._f_A.shape[1]
        res = bootstrap([np.arange(n)], statistic=statistic, method='BCa', n_resamples=n_resamples, confidence_level=confidence_level, bootstrap_result=self._bootstrap_result)
        self._bootstrap_result = res
        first_order = BootstrapResult(confidence_interval=ConfidenceInterval(res.confidence_interval.low[0], res.confidence_interval.high[0]), bootstrap_distribution=res.bootstrap_distribution[0], standard_error=res.standard_error[0])
        total_order = BootstrapResult(confidence_interval=ConfidenceInterval(res.confidence_interval.low[1], res.confidence_interval.high[1]), bootstrap_distribution=res.bootstrap_distribution[1], standard_error=res.standard_error[1])
        return BootstrapSobolResult(first_order=first_order, total_order=total_order)
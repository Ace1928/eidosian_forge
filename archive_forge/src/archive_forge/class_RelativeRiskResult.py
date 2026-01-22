import operator
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri
from ._common import ConfidenceInterval
@dataclass
class RelativeRiskResult:
    """
    Result of `scipy.stats.contingency.relative_risk`.

    Attributes
    ----------
    relative_risk : float
        This is::

            (exposed_cases/exposed_total) / (control_cases/control_total)

    exposed_cases : int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : int
        The total number of "exposed" individuals in the sample.
    control_cases : int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : int
        The total number of "control" individuals in the sample.

    Methods
    -------
    confidence_interval :
        Compute the confidence interval for the relative risk estimate.
    """
    relative_risk: float
    exposed_cases: int
    exposed_total: int
    control_cases: int
    control_total: int

    def confidence_interval(self, confidence_level=0.95):
        """
        Compute the confidence interval for the relative risk.

        The confidence interval is computed using the Katz method
        (i.e. "Method C" of [1]_; see also [2]_, section 3.1.2).

        Parameters
        ----------
        confidence_level : float, optional
            The confidence level to use for the confidence interval.
            Default is 0.95.

        Returns
        -------
        ci : ConfidenceInterval instance
            The return value is an object with attributes ``low`` and
            ``high`` that hold the confidence interval.

        References
        ----------
        .. [1] D. Katz, J. Baptista, S. P. Azen and M. C. Pike, "Obtaining
               confidence intervals for the risk ratio in cohort studies",
               Biometrics, 34, 469-474 (1978).
        .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
               CRC Press LLC, Boca Raton, FL, USA (1996).


        Examples
        --------
        >>> from scipy.stats.contingency import relative_risk
        >>> result = relative_risk(exposed_cases=10, exposed_total=75,
        ...                        control_cases=12, control_total=225)
        >>> result.relative_risk
        2.5
        >>> result.confidence_interval()
        ConfidenceInterval(low=1.1261564003469628, high=5.549850800541033)
        """
        if not 0 <= confidence_level <= 1:
            raise ValueError('confidence_level must be in the interval [0, 1].')
        if self.exposed_cases == 0 and self.control_cases == 0:
            return ConfidenceInterval(low=np.nan, high=np.nan)
        elif self.exposed_cases == 0:
            return ConfidenceInterval(low=0.0, high=np.nan)
        elif self.control_cases == 0:
            return ConfidenceInterval(low=np.nan, high=np.inf)
        alpha = 1 - confidence_level
        z = ndtri(1 - alpha / 2)
        rr = self.relative_risk
        se = np.sqrt(1 / self.exposed_cases - 1 / self.exposed_total + 1 / self.control_cases - 1 / self.control_total)
        delta = z * se
        katz_lo = rr * np.exp(-delta)
        katz_hi = rr * np.exp(delta)
        return ConfidenceInterval(low=katz_lo, high=katz_hi)
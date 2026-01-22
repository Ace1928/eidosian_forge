import operator
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri
from ._common import ConfidenceInterval

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
        
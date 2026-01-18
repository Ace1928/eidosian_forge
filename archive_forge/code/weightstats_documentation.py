import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly

        test of equivalence for two independent samples, based on z-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        
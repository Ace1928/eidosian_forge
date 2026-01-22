import numpy as np
from scipy import stats
import pandas as pd
class PredictionResultsBase:
    """Based class for get_prediction results
    """

    def __init__(self, predicted, var_pred, func=None, deriv=None, df=None, dist=None, row_labels=None, **kwds):
        self.predicted = predicted
        self.var_pred = var_pred
        self.func = func
        self.deriv = deriv
        self.df = df
        self.row_labels = row_labels
        self.__dict__.update(kwds)
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se(self):
        return np.sqrt(self.var_pred)

    @property
    def tvalues(self):
        return self.predicted / self.se

    def t_test(self, value=0, alternative='two-sided'):
        """z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.

        """
        stat = (self.predicted - value) / self.se
        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args) * 2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return (stat, pvalue)

    def _conf_int_generic(self, center, se, alpha, dist_args=None):
        """internal function to avoid code duplication
        """
        if dist_args is None:
            dist_args = ()
        q = self.dist.ppf(1 - alpha / 2.0, *dist_args)
        lower = center - q * se
        upper = center + q * se
        ci = np.column_stack((lower, upper))
        return ci

    def conf_int(self, *, alpha=0.05, **kwds):
        """Confidence interval for the predicted value.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        kwds : extra keyword arguments
            Ignored in base class, only for compatibility, consistent signature
            with subclasses

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        ci = self._conf_int_generic(self.predicted, self.se, alpha, dist_args=self.dist_args)
        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pandas DataFrame with columns 'predicted', 'se', 'ci_lower', 'ci_upper'
        """
        ci = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['predicted'] = self.predicted
        to_include['se'] = self.se
        to_include['ci_lower'] = ci[:, 0]
        to_include['ci_upper'] = ci[:, 1]
        self.table = to_include
        res = pd.DataFrame(to_include, index=self.row_labels, columns=to_include.keys())
        return res
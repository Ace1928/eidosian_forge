import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
class Table2x2(SquareTable):
    """
    Analyses that can be performed on a 2x2 contingency table.

    Parameters
    ----------
    table : array_like
        A 2x2 contingency table
    shift_zeros : bool
        If true, 0.5 is added to all cells of the table if any cell is
        equal to zero.

    Notes
    -----
    The inference procedures used here are all based on a sampling
    model in which the units are independent and identically
    distributed, with each unit being classified with respect to two
    categorical variables.

    Note that for the risk ratio, the analysis is not symmetric with
    respect to the rows and columns of the contingency table.  The two
    rows define population subgroups, column 0 is the number of
    'events', and column 1 is the number of 'non-events'.
    """

    def __init__(self, table, shift_zeros=True):
        if type(table) is list:
            table = np.asarray(table)
        if table.ndim != 2 or table.shape[0] != 2 or table.shape[1] != 2:
            raise ValueError('Table2x2 takes a 2x2 table as input.')
        super().__init__(table, shift_zeros)

    @classmethod
    def from_data(cls, data, shift_zeros=True):
        """
        Construct a Table object from data.

        Parameters
        ----------
        data : array_like
            The raw data, the first column defines the rows and the
            second column defines the columns.
        shift_zeros : bool
            If True, and if there are any zeros in the contingency
            table, add 0.5 to all four cells of the table.
        """
        if isinstance(data, pd.DataFrame):
            table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
        else:
            table = pd.crosstab(data[:, 0], data[:, 1])
        return cls(table, shift_zeros)

    @cache_readonly
    def log_oddsratio(self):
        """
        Returns the log odds ratio for a 2x2 table.
        """
        f = self.table.flatten()
        return np.dot(np.log(f), np.r_[1, -1, -1, 1])

    @cache_readonly
    def oddsratio(self):
        """
        Returns the odds ratio for a 2x2 table.
        """
        return self.table[0, 0] * self.table[1, 1] / (self.table[0, 1] * self.table[1, 0])

    @cache_readonly
    def log_oddsratio_se(self):
        """
        Returns the standard error for the log odds ratio.
        """
        return np.sqrt(np.sum(1 / self.table))

    def oddsratio_pvalue(self, null=1):
        """
        P-value for a hypothesis test about the odds ratio.

        Parameters
        ----------
        null : float
            The null value of the odds ratio.
        """
        return self.log_oddsratio_pvalue(np.log(null))

    def log_oddsratio_pvalue(self, null=0):
        """
        P-value for a hypothesis test about the log odds ratio.

        Parameters
        ----------
        null : float
            The null value of the log odds ratio.
        """
        zscore = (self.log_oddsratio - null) / self.log_oddsratio_se
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
        return pvalue

    def log_oddsratio_confint(self, alpha=0.05, method='normal'):
        """
        A confidence level for the log odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        f = -stats.norm.ppf(alpha / 2)
        lor = self.log_oddsratio
        se = self.log_oddsratio_se
        lcb = lor - f * se
        ucb = lor + f * se
        return (lcb, ucb)

    def oddsratio_confint(self, alpha=0.05, method='normal'):
        """
        A confidence interval for the odds ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        lcb, ucb = self.log_oddsratio_confint(alpha, method=method)
        return (np.exp(lcb), np.exp(ucb))

    @cache_readonly
    def riskratio(self):
        """
        Returns the risk ratio for a 2x2 table.

        The risk ratio is calculated with respect to the rows.
        """
        p = self.table[:, 0] / self.table.sum(1)
        return p[0] / p[1]

    @cache_readonly
    def log_riskratio(self):
        """
        Returns the log of the risk ratio.
        """
        return np.log(self.riskratio)

    @cache_readonly
    def log_riskratio_se(self):
        """
        Returns the standard error of the log of the risk ratio.
        """
        n = self.table.sum(1)
        p = self.table[:, 0] / n
        va = np.sum((1 - p) / (n * p))
        return np.sqrt(va)

    def riskratio_pvalue(self, null=1):
        """
        p-value for a hypothesis test about the risk ratio.

        Parameters
        ----------
        null : float
            The null value of the risk ratio.
        """
        return self.log_riskratio_pvalue(np.log(null))

    def log_riskratio_pvalue(self, null=0):
        """
        p-value for a hypothesis test about the log risk ratio.

        Parameters
        ----------
        null : float
            The null value of the log risk ratio.
        """
        zscore = (self.log_riskratio - null) / self.log_riskratio_se
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
        return pvalue

    def log_riskratio_confint(self, alpha=0.05, method='normal'):
        """
        A confidence interval for the log risk ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        f = -stats.norm.ppf(alpha / 2)
        lrr = self.log_riskratio
        se = self.log_riskratio_se
        lcb = lrr - f * se
        ucb = lrr + f * se
        return (lcb, ucb)

    def riskratio_confint(self, alpha=0.05, method='normal'):
        """
        A confidence interval for the risk ratio.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the
            confidence interval.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        lcb, ucb = self.log_riskratio_confint(alpha, method=method)
        return (np.exp(lcb), np.exp(ucb))

    def summary(self, alpha=0.05, float_format='%.3f', method='normal'):
        """
        Summarizes results for a 2x2 table analysis.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the confidence
            intervals.
        float_format : str
            Used to format the numeric values in the table.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """

        def fmt(x):
            if isinstance(x, str):
                return x
            return float_format % x
        headers = ['Estimate', 'SE', 'LCB', 'UCB', 'p-value']
        stubs = ['Odds ratio', 'Log odds ratio', 'Risk ratio', 'Log risk ratio']
        lcb1, ucb1 = self.oddsratio_confint(alpha, method)
        lcb2, ucb2 = self.log_oddsratio_confint(alpha, method)
        lcb3, ucb3 = self.riskratio_confint(alpha, method)
        lcb4, ucb4 = self.log_riskratio_confint(alpha, method)
        data = [[fmt(x) for x in [self.oddsratio, '', lcb1, ucb1, self.oddsratio_pvalue()]], [fmt(x) for x in [self.log_oddsratio, self.log_oddsratio_se, lcb2, ucb2, self.oddsratio_pvalue()]], [fmt(x) for x in [self.riskratio, '', lcb3, ucb3, self.riskratio_pvalue()]], [fmt(x) for x in [self.log_riskratio, self.log_riskratio_se, lcb4, ucb4, self.riskratio_pvalue()]]]
        tab = iolib.SimpleTable(data, headers, stubs, data_aligns='r', table_dec_above='')
        return tab
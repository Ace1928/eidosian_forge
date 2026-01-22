import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
class MultivariateTestResults:
    """
    Multivariate test results class

    Returned by `mv_test` method of `_MultivariateOLSResults` class

    Parameters
    ----------
    results : dict[str, dict]
        Dictionary containing test results. See the description
        below for the expected format.
    endog_names : sequence[str]
        A list or other sequence of endogenous variables names
    exog_names : sequence[str]
        A list of other sequence of exogenous variables names

    Attributes
    ----------
    results : dict
        Each hypothesis is contained in a single`key`. Each test must
        have the following keys:

        * 'stat' - contains the multivariate test results
        * 'contrast_L' - contains the contrast_L matrix
        * 'transform_M' - contains the transform_M matrix
        * 'constant_C' - contains the constant_C matrix
        * 'H' - contains an intermediate Hypothesis matrix,
          or the between groups sums of squares and cross-products matrix,
          corresponding to the numerator of the univariate F test.
        * 'E' - contains an intermediate Error matrix,
          corresponding to the denominator of the univariate F test.
          The Hypotheses and Error matrices can be used to calculate
          the same test statistics in 'stat', as well as to calculate
          the discriminant function (canonical correlates) from the
          eigenvectors of inv(E)H.

    endog_names : list[str]
        The endogenous names
    exog_names : list[str]
        The exogenous names
    summary_frame : DataFrame
        Returns results as a MultiIndex DataFrame
    """

    def __init__(self, results, endog_names, exog_names):
        self.results = results
        self.endog_names = list(endog_names)
        self.exog_names = list(exog_names)

    def __str__(self):
        return self.summary().__str__()

    def __getitem__(self, item):
        return self.results[item]

    @property
    def summary_frame(self):
        """
        Return results as a multiindex dataframe
        """
        df = []
        for key in self.results:
            tmp = self.results[key]['stat'].copy()
            tmp.loc[:, 'Effect'] = key
            df.append(tmp.reset_index())
        df = pd.concat(df, axis=0)
        df = df.set_index(['Effect', 'index'])
        df.index.set_names(['Effect', 'Statistic'], inplace=True)
        return df

    def summary(self, show_contrast_L=False, show_transform_M=False, show_constant_C=False):
        """
        Summary of test results

        Parameters
        ----------
        show_contrast_L : bool
            Whether to show contrast_L matrix
        show_transform_M : bool
            Whether to show transform_M matrix
        show_constant_C : bool
            Whether to show the constant_C
        """
        summ = summary2.Summary()
        summ.add_title('Multivariate linear model')
        for key in self.results:
            summ.add_dict({'': ''})
            df = self.results[key]['stat'].copy()
            df = df.reset_index()
            c = list(df.columns)
            c[0] = key
            df.columns = c
            df.index = ['', '', '', '']
            summ.add_df(df)
            if show_contrast_L:
                summ.add_dict({key: ' contrast L='})
                df = pd.DataFrame(self.results[key]['contrast_L'], columns=self.exog_names)
                summ.add_df(df)
            if show_transform_M:
                summ.add_dict({key: ' transform M='})
                df = pd.DataFrame(self.results[key]['transform_M'], index=self.endog_names)
                summ.add_df(df)
            if show_constant_C:
                summ.add_dict({key: ' constant C='})
                df = pd.DataFrame(self.results[key]['constant_C'])
                summ.add_df(df)
        return summ
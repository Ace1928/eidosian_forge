from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
class AnovaRM:
    """
    Repeated measures Anova using least squares regression

    The full model regression residual sum of squares is
    used to compare with the reduced model for calculating the
    within-subject effect sum of squares [1].

    Currently, only fully balanced within-subject designs are supported.
    Calculation of between-subject effects and corrections for violation of
    sphericity are not yet implemented.

    Parameters
    ----------
    data : DataFrame
    depvar : str
        The dependent variable in `data`
    subject : str
        Specify the subject id
    within : list[str]
        The within-subject factors
    between : list[str]
        The between-subject factors, this is not yet implemented
    aggregate_func : {None, 'mean', callable}
        If the data set contains more than a single observation per subject
        and cell of the specified model, this function will be used to
        aggregate the data before running the Anova. `None` (the default) will
        not perform any aggregation; 'mean' is s shortcut to `numpy.mean`.
        An exception will be raised if aggregation is required, but no
        aggregation function was specified.

    Returns
    -------
    results : AnovaResults instance

    Raises
    ------
    ValueError
        If the data need to be aggregated, but `aggregate_func` was not
        specified.

    Notes
    -----
    This implementation currently only supports fully balanced designs. If the
    data contain more than one observation per subject and cell of the design,
    these observations need to be aggregated into a single observation
    before the Anova is calculated, either manually or by passing an aggregation
    function via the `aggregate_func` keyword argument.
    Note that if the input data set was not balanced before performing the
    aggregation, the implied heteroscedasticity of the data is ignored.

    References
    ----------
    .. [*] Rutherford, Andrew. Anova and ANCOVA: a GLM approach. John Wiley & Sons, 2011.
    """

    def __init__(self, data, depvar, subject, within=None, between=None, aggregate_func=None):
        self.data = data
        self.depvar = depvar
        self.within = within
        if 'C' in within:
            raise ValueError("Factor name cannot be 'C'! This is in conflict with patsy's contrast function name.")
        self.between = between
        if between is not None:
            raise NotImplementedError('Between subject effect not yet supported!')
        self.subject = subject
        if aggregate_func == 'mean':
            self.aggregate_func = pd.Series.mean
        else:
            self.aggregate_func = aggregate_func
        if not data.equals(data.drop_duplicates(subset=[subject] + within)):
            if self.aggregate_func is not None:
                self._aggregate()
            else:
                msg = 'The data set contains more than one observation per subject and cell. Either aggregate the data manually, or pass the `aggregate_func` parameter.'
                raise ValueError(msg)
        self._check_data_balanced()

    def _aggregate(self):
        self.data = self.data.groupby([self.subject] + self.within, as_index=False)[self.depvar].agg(self.aggregate_func)

    def _check_data_balanced(self):
        """raise if data is not balanced

        This raises a ValueError if the data is not balanced, and
        returns None if it is balance

        Return might change
        """
        factor_levels = 1
        for wi in self.within:
            factor_levels *= len(self.data[wi].unique())
        cell_count = {}
        for index in range(self.data.shape[0]):
            key = []
            for col in self.within:
                key.append(self.data[col].iloc[index])
            key = tuple(key)
            if key in cell_count:
                cell_count[key] = cell_count[key] + 1
            else:
                cell_count[key] = 1
        error_message = 'Data is unbalanced.'
        if len(cell_count) != factor_levels:
            raise ValueError(error_message)
        count = cell_count[key]
        for key in cell_count:
            if count != cell_count[key]:
                raise ValueError(error_message)
        if self.data.shape[0] > count * factor_levels:
            raise ValueError('There are more than 1 element in a cell! Missing factors?')

    def fit(self):
        """estimate the model and compute the Anova table

        Returns
        -------
        AnovaResults instance
        """
        y = self.data[self.depvar].values
        within = ['C(%s, Sum)' % i for i in self.within]
        subject = 'C(%s, Sum)' % self.subject
        factors = within + [subject]
        x = patsy.dmatrix('*'.join(factors), data=self.data)
        term_slices = x.design_info.term_name_slices
        for key in term_slices:
            ind = np.array([False] * x.shape[1])
            ind[term_slices[key]] = True
            term_slices[key] = np.array(ind)
        term_exclude = [':'.join(factors)]
        ind = _not_slice(term_slices, term_exclude, x.shape[1])
        x = x[:, ind]
        model = OLS(y, x)
        results = model.fit()
        if model.rank < x.shape[1]:
            raise ValueError('Independent variables are collinear.')
        for i in term_exclude:
            term_slices.pop(i)
        for key in term_slices:
            term_slices[key] = term_slices[key][ind]
        params = results.params
        df_resid = results.df_resid
        ssr = results.ssr
        columns = ['F Value', 'Num DF', 'Den DF', 'Pr > F']
        anova_table = pd.DataFrame(np.zeros((0, 4)), columns=columns)
        for key in term_slices:
            if self.subject not in key and key != 'Intercept':
                ssr1, df_resid1 = _ssr_reduced_model(y, x, term_slices, params, [key])
                df1 = df_resid1 - df_resid
                msm = (ssr1 - ssr) / df1
                if key == ':'.join(factors[:-1]) or key + ':' + subject not in term_slices:
                    mse = ssr / df_resid
                    df2 = df_resid
                else:
                    ssr1, df_resid1 = _ssr_reduced_model(y, x, term_slices, params, [key + ':' + subject])
                    df2 = df_resid1 - df_resid
                    mse = (ssr1 - ssr) / df2
                F = msm / mse
                p = stats.f.sf(F, df1, df2)
                term = key.replace('C(', '').replace(', Sum)', '')
                anova_table.loc[term, 'F Value'] = F
                anova_table.loc[term, 'Num DF'] = df1
                anova_table.loc[term, 'Den DF'] = df2
                anova_table.loc[term, 'Pr > F'] = p
        return AnovaResults(anova_table)
import pandas as pd
import patsy
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender
def _make_exog_from_arrays(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit without a formula.

    Returns
    -------
    exog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    """
    model = result.model
    model_exog = model.exog
    exog_names = model.exog_names
    if summaries is None:
        summaries = {}
    if values is None:
        values = {}
    exog = np.zeros((num_points, model_exog.shape[1]))
    colnames = list(values.keys()) + list(summaries.keys()) + [focus_var]
    unmatched = set(exog_names) - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn('%s in model but not in `summaries` or `values`.' % ', '.join(["'%s'" % x for x in unmatched]), ValueWarning)
    pctls = np.linspace(0, 100, num_points).tolist()
    ix = exog_names.index(focus_var)
    fvals = np.percentile(model_exog[:, ix], pctls)
    exog[:, ix] = fvals
    for ky in summaries.keys():
        ix = exog_names.index(ky)
        exog[:, ix] = summaries[ky](model_exog[:, ix])
    for ky in values.keys():
        ix = exog_names.index(ky)
        exog[:, ix] = values[ky]
    return (exog, fvals)
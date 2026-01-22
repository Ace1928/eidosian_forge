from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
class OrdinalGEEResults(GEEResults):
    __doc__ = 'This class summarizes the fit of a marginal regression modelfor an ordinal response using GEE.\n' + _gee_results_doc

    def plot_distribution(self, ax=None, exog_values=None):
        """
        Plot the fitted probabilities of endog in an ordinal model,
        for specified values of the predictors.

        Parameters
        ----------
        ax : AxesSubplot
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        exog_values : array_like
            A list of dictionaries, with each dictionary mapping
            variable names to values at which the variable is held
            fixed.  The values P(endog=y | exog) are plotted for all
            possible values of y, at the given exog value.  Variables
            not included in a dictionary are held fixed at the mean
            value.

        Example:
        --------
        We have a model with covariates 'age' and 'sex', and wish to
        plot the probabilities P(endog=y | exog) for males (sex=0) and
        for females (sex=1), as separate paths on the plot.  Since
        'age' is not included below in the map, it is held fixed at
        its mean value.

        >>> ev = [{"sex": 1}, {"sex": 0}]
        >>> rslt.distribution_plot(exog_values=ev)
        """
        from statsmodels.graphics import utils as gutils
        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()
        if exog_values is None:
            exog_values = [{}]
        exog_means = self.model.exog.mean(0)
        ix_icept = [i for i, x in enumerate(self.model.exog_names) if x.startswith('I(')]
        for ev in exog_values:
            for k in ev.keys():
                if k not in self.model.exog_names:
                    raise ValueError('%s is not a variable in the model' % k)
            pr = []
            for j in ix_icept:
                xp = np.zeros_like(self.params)
                xp[j] = 1.0
                for i, vn in enumerate(self.model.exog_names):
                    if i in ix_icept:
                        continue
                    if vn in ev:
                        xp[i] = ev[vn]
                    else:
                        xp[i] = exog_means[i]
                p = 1 / (1 + np.exp(-np.dot(xp, self.params)))
                pr.append(p)
            pr.insert(0, 1)
            pr.append(0)
            pr = np.asarray(pr)
            prd = -np.diff(pr)
            ax.plot(self.model.endog_values, prd, 'o-')
        ax.set_xlabel('Response value')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        return fig
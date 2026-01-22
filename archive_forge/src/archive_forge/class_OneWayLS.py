import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS
class OneWayLS:
    """Class to test equality of regression coefficients across groups

    This class performs tests whether the linear regression coefficients are
    the same across pre-specified groups. This can be used to test for
    structural breaks at given change points, or for ANOVA style analysis of
    differences in the effect of explanatory variables across groups.

    Notes
    -----
    The test is implemented by regression on the original pooled exogenous
    variables and on group dummies times the exogenous regressors.

    y_i = X_i beta_i + u_i  for all groups i

    The test is for the null hypothesis: beta_i = beta for all i
    against the alternative that at least one beta_i is different.

    By default it is assumed that all u_i have the same variance. If the
    keyword option het is True, then it is assumed that the variance is
    group specific. This uses WLS with weights given by the standard errors
    from separate regressions for each group.
    Note: het=True is not sufficiently tested

    The F-test assumes that the errors are normally distributed.



    original question from mailing list for equality of coefficients
    across regressions, and example in Stata FAQ

    *testing*:

    * if constant is the only regressor then the result for the F-test is
      the same as scipy.stats.f_oneway
      (which in turn is verified against NIST for not badly scaled problems)
    * f-test for simple structural break is the same as in original script
    * power and size of test look ok in examples
    * not checked/verified for heteroskedastic case
      - for constant only: ftest result is the same with WLS as with OLS - check?

    check: I might be mixing up group names (unique)
           and group id (integers in arange(ngroups)
           not tested for groups that are not arange(ngroups)
           make sure groupnames are always consistently sorted/ordered
           Fixed for getting the results, but groups are not printed yet, still
           inconsistent use for summaries of results.
    """

    def __init__(self, y, x, groups=None, het=False, data=None, meta=None):
        if groups is None:
            raise ValueError('use OLS if there are no groups')
        if data:
            y = data[y]
            x = [data[v] for v in x]
            try:
                groups = data[groups]
            except [KeyError, ValueError]:
                pass
        self.endog = np.asarray(y)
        self.exog = np.asarray(x)
        if self.exog.ndim == 1:
            self.exog = self.exog[:, None]
        self.groups = np.asarray(groups)
        self.het = het
        self.groupsint = None
        if np.issubdtype(self.groups.dtype, int):
            self.unique = np.unique(self.groups)
            if (self.unique == np.arange(len(self.unique))).all():
                self.groupsint = self.groups
        if self.groupsint is None:
            self.unique, self.groupsint = np.unique(self.groups, return_inverse=True)
        self.uniqueint = np.arange(len(self.unique))

    def fitbygroups(self):
        """Fit OLS regression for each group separately.

        Returns
        -------
        results are attached

        olsbygroup : dictionary of result instance
            the returned regression results for each group
        sigmabygroup : array (ngroups,) (this should be called sigma2group ??? check)
            mse_resid for each group
        weights : array (nobs,)
            standard deviation of group extended to the original observations. This can
            be used as weights in WLS for group-wise heteroscedasticity.



        """
        olsbygroup = {}
        sigmabygroup = []
        for gi, group in enumerate(self.unique):
            groupmask = self.groupsint == gi
            res = OLS(self.endog[groupmask], self.exog[groupmask]).fit()
            olsbygroup[group] = res
            sigmabygroup.append(res.mse_resid)
        self.olsbygroup = olsbygroup
        self.sigmabygroup = np.array(sigmabygroup)
        self.weights = np.sqrt(self.sigmabygroup[self.groupsint])

    def fitjoint(self):
        """fit a joint fixed effects model to all observations

        The regression results are attached as `lsjoint`.

        The contrasts for overall and pairwise tests for equality of coefficients are
        attached as a dictionary `contrasts`. This also includes the contrasts for the test
        that the coefficients of a level are zero. ::

        >>> res.contrasts.keys()
        [(0, 1), 1, 'all', 3, (1, 2), 2, (1, 3), (2, 3), (0, 3), (0, 2)]

        The keys are based on the original names or labels of the groups.

        TODO: keys can be numpy scalars and then the keys cannot be sorted



        """
        if not hasattr(self, 'weights'):
            self.fitbygroups()
        groupdummy = (self.groupsint[:, None] == self.uniqueint).astype(int)
        dummyexog = self.exog[:, None, :] * groupdummy[:, 1:, None]
        exog = np.c_[self.exog, dummyexog.reshape(self.exog.shape[0], -1)]
        if self.het:
            weights = self.weights
            res = WLS(self.endog, exog, weights=weights).fit()
        else:
            res = OLS(self.endog, exog).fit()
        self.lsjoint = res
        contrasts = {}
        nvars = self.exog.shape[1]
        nparams = exog.shape[1]
        ndummies = nparams - nvars
        contrasts['all'] = np.c_[np.zeros((ndummies, nvars)), np.eye(ndummies)]
        for groupind, group in enumerate(self.unique[1:]):
            groupind = groupind + 1
            contr = np.zeros((nvars, nparams))
            contr[:, nvars * groupind:nvars * (groupind + 1)] = np.eye(nvars)
            contrasts[group] = contr
            contrasts[self.unique[0], group] = contr
        pairs = np.triu_indices(len(self.unique), 1)
        for ind1, ind2 in zip(*pairs):
            if ind1 == 0:
                continue
            g1 = self.unique[ind1]
            g2 = self.unique[ind2]
            group = (g1, g2)
            contr = np.zeros((nvars, nparams))
            contr[:, nvars * ind1:nvars * (ind1 + 1)] = np.eye(nvars)
            contr[:, nvars * ind2:nvars * (ind2 + 1)] = -np.eye(nvars)
            contrasts[group] = contr
        self.contrasts = contrasts

    def fitpooled(self):
        """fit the pooled model, which assumes there are no differences across groups
        """
        if self.het:
            if not hasattr(self, 'weights'):
                self.fitbygroups()
            weights = self.weights
            res = WLS(self.endog, self.exog, weights=weights).fit()
        else:
            res = OLS(self.endog, self.exog).fit()
        self.lspooled = res

    def ftest_summary(self):
        """run all ftests on the joint model

        Returns
        -------
        fres : str
           a string that lists the results of all individual f-tests
        summarytable : list of tuples
           contains (pair, (fvalue, pvalue,df_denom, df_num)) for each f-test

        Note
        ----
        This are the raw results and not formatted for nice printing.

        """
        if not hasattr(self, 'lsjoint'):
            self.fitjoint()
        txt = []
        summarytable = []
        txt.append('F-test for equality of coefficients across groups')
        fres = self.lsjoint.f_test(self.contrasts['all'])
        txt.append(fres.__str__())
        summarytable.append(('all', (fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))
        pairs = np.triu_indices(len(self.unique), 1)
        for ind1, ind2 in zip(*pairs):
            g1 = self.unique[ind1]
            g2 = self.unique[ind2]
            txt.append('F-test for equality of coefficients between group %s and group %s' % (g1, g2))
            group = (g1, g2)
            fres = self.lsjoint.f_test(self.contrasts[group])
            txt.append(fres.__str__())
            summarytable.append((group, (fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))
        self.summarytable = summarytable
        return ('\n'.join(txt), summarytable)

    def print_summary(self, res):
        """printable string of summary

        """
        groupind = res.groups
        if hasattr(res, 'self.summarytable'):
            summtable = self.summarytable
        else:
            _, summtable = res.ftest_summary()
        txt = ''
        templ = "Table of F-tests for overall or pairwise equality of coefficients'\n%(tab)s\n\n\nNotes: p-values are not corrected for many tests\n       (no Bonferroni correction)\n       * : reject at 5%% uncorrected confidence level\nNull hypothesis: all or pairwise coefficient are the same'\nAlternative hypothesis: all coefficients are different'\n\n\nComparison with stats.f_oneway\n%(statsfow)s\n\n\nLikelihood Ratio Test\n%(lrtest)s\nNull model: pooled all coefficients are the same across groups,'\nAlternative model: all coefficients are allowed to be different'\nnot verified but looks close to f-test result'\n\n\nOLS parameters by group from individual, separate ols regressions'\n%(olsbg)s\nfor group in sorted(res.olsbygroup):\n    r = res.olsbygroup[group]\n    print group, r.params\n\n\nCheck for heteroscedasticity, '\nvariance and standard deviation for individual regressions'\n%(grh)s\nvariance    ', res.sigmabygroup\nstandard dev', np.sqrt(res.sigmabygroup)\n"
        from statsmodels.iolib import SimpleTable
        resvals = {}
        resvals['tab'] = str(SimpleTable([['{!r}'.format(row[0])] + list(row[1]) + ['*'] * (row[1][1] > 0.5).item() for row in summtable], headers=['pair', 'F-statistic', 'p-value', 'df_denom', 'df_num']))
        resvals['statsfow'] = str(stats.f_oneway(*[res.endog[groupind == gr] for gr in res.unique]))
        resvals['lrtest'] = str(SimpleTable([res.lr_test()], headers=['likelihood ratio', 'p-value', 'df']))
        resvals['olsbg'] = str(SimpleTable([[group] + res.olsbygroup[group].params.tolist() for group in sorted(res.olsbygroup)]))
        resvals['grh'] = str(SimpleTable(np.vstack([res.sigmabygroup, np.sqrt(res.sigmabygroup)]), headers=res.unique.tolist()))
        return templ % resvals

    def lr_test(self):
        """
        generic likelihood ratio test between nested models

            \\begin{align}
            D & = -2(\\ln(\\text{likelihood for null model}) - \\ln(\\text{likelihood for alternative model})) \\\\
            & = -2\\ln\\left( \\frac{\\text{likelihood for null model}}{\\text{likelihood for alternative model}} \\right).
            \\end{align}

        is distributed as chisquare with df equal to difference in number of parameters or equivalently
        difference in residual degrees of freedom  (sign?)

        TODO: put into separate function
        """
        if not hasattr(self, 'lsjoint'):
            self.fitjoint()
        if not hasattr(self, 'lspooled'):
            self.fitpooled()
        loglikejoint = self.lsjoint.llf
        loglikepooled = self.lspooled.llf
        lrstat = -2 * (loglikepooled - loglikejoint)
        lrdf = self.lspooled.df_resid - self.lsjoint.df_resid
        lrpval = stats.chi2.sf(lrstat, lrdf)
        return (lrstat, lrpval, lrdf)
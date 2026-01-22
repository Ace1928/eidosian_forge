import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
class TestRegCoefD(TestRegCoefC):
    """
    Significance test for the categorical variables in a nonparametric
    regression.

    Parameters
    ----------
    model : Instance of KernelReg class
        This is the nonparametric regression model whose elements
        are tested for significance.
    test_vars : tuple, list of one element
        index of position of the discrete variable to be tested
        for significance. E.g. (3) tests variable at
        position 3 for significance.
    nboot : int
        Number of bootstrap samples used to determine the distribution
        of the test statistic in a finite sample. Default is 400

    Attributes
    ----------
    sig : str
        The significance level of the variable(s) tested
        "Not Significant": Not significant at the 90% confidence level
                            Fails to reject the null
        "*": Significant at the 90% confidence level
        "**": Significant at the 95% confidence level
        "***": Significant at the 99% confidence level

    Notes
    -----
    This class currently does not allow joint hypothesis.
    Only one variable can be tested at a time

    References
    ----------
    See [9] and chapter 12 in [1].
    """

    def _compute_test_stat(self, Y, X):
        """Computes the test statistic"""
        dom_x = np.sort(np.unique(self.exog[:, self.test_vars]))
        n = np.shape(X)[0]
        model = KernelReg(Y, X, self.var_type, self.model.reg_type, self.bw, defaults=EstimatorSettings(efficient=False))
        X1 = copy.deepcopy(X)
        X1[:, self.test_vars] = 0
        m0 = model.fit(data_predict=X1)[0]
        m0 = np.reshape(m0, (n, 1))
        zvec = np.zeros((n, 1))
        for i in dom_x[1:]:
            X1[:, self.test_vars] = i
            m1 = model.fit(data_predict=X1)[0]
            m1 = np.reshape(m1, (n, 1))
            zvec += (m1 - m0) ** 2
        avg = zvec.sum(axis=0) / float(n)
        return avg

    def _compute_sig(self):
        """Calculates the significance level of the variable tested"""
        m = self._est_cond_mean()
        Y = self.endog
        X = self.exog
        n = np.shape(X)[0]
        u = Y - m
        u = u - np.mean(u)
        fct1 = (1 - 5 ** 0.5) / 2.0
        fct2 = (1 + 5 ** 0.5) / 2.0
        u1 = fct1 * u
        u2 = fct2 * u
        r = fct2 / 5 ** 0.5
        I_dist = np.empty((self.nboot, 1))
        for j in range(self.nboot):
            u_boot = copy.deepcopy(u2)
            prob = np.random.uniform(0, 1, size=(n, 1))
            ind = prob < r
            u_boot[ind] = u1[ind]
            Y_boot = m + u_boot
            I_dist[j] = self._compute_test_stat(Y_boot, X)
        sig = 'Not Significant'
        if self.test_stat > mquantiles(I_dist, 0.9):
            sig = '*'
        if self.test_stat > mquantiles(I_dist, 0.95):
            sig = '**'
        if self.test_stat > mquantiles(I_dist, 0.99):
            sig = '***'
        return sig

    def _est_cond_mean(self):
        """
        Calculates the expected conditional mean
        m(X, Z=l) for all possible l
        """
        self.dom_x = np.sort(np.unique(self.exog[:, self.test_vars]))
        X = copy.deepcopy(self.exog)
        m = 0
        for i in self.dom_x:
            X[:, self.test_vars] = i
            m += self.model.fit(data_predict=X)[0]
        m = m / float(len(self.dom_x))
        m = np.reshape(m, (np.shape(self.exog)[0], 1))
        return m
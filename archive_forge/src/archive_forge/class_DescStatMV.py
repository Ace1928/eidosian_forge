import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
class DescStatMV(_OptFuncts):
    """
    A class for conducting inference on multivariate means and correlation.

    Parameters
    ----------
    endog : ndarray
        Data to be analyzed

    Attributes
    ----------
    endog : ndarray
        Data to be analyzed

    nobs : float
        Number of observations
    """

    def __init__(self, endog):
        self.endog = endog
        self.nobs = endog.shape[0]

    def mv_test_mean(self, mu_array, return_weights=False):
        """
        Returns -2 x log likelihood and the p-value
        for a multivariate hypothesis test of the mean

        Parameters
        ----------
        mu_array  : 1d array
            Hypothesized values for the mean.  Must have same number of
            elements as columns in endog

        return_weights : bool
            If True, returns the weights that maximize the
            likelihood of mu_array. Default is False.

        Returns
        -------
        test_results : tuple
            The log-likelihood ratio and p-value for mu_array
        """
        endog = self.endog
        nobs = self.nobs
        if len(mu_array) != endog.shape[1]:
            raise ValueError('mu_array must have the same number of elements as the columns of the data.')
        mu_array = mu_array.reshape(1, endog.shape[1])
        means = np.ones((endog.shape[0], endog.shape[1]))
        means = mu_array * means
        est_vect = endog - means
        start_vals = 1.0 / nobs * np.ones(endog.shape[1])
        eta_star = self._modif_newton(start_vals, est_vect, np.ones(nobs) * (1.0 / nobs))
        denom = 1 + np.dot(eta_star, est_vect.T)
        self.new_weights = 1 / nobs * 1 / denom
        llr = -2 * np.sum(np.log(nobs * self.new_weights))
        p_val = chi2.sf(llr, mu_array.shape[1])
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        else:
            return (llr, p_val)

    def mv_mean_contour(self, mu1_low, mu1_upp, mu2_low, mu2_upp, step1, step2, levs=(0.001, 0.01, 0.05, 0.1, 0.2), var1_name=None, var2_name=None, plot_dta=False):
        """
        Creates a confidence region plot for the mean of bivariate data

        Parameters
        ----------
        m1_low : float
            Minimum value of the mean for variable 1
        m1_upp : float
            Maximum value of the mean for variable 1
        mu2_low : float
            Minimum value of the mean for variable 2
        mu2_upp : float
            Maximum value of the mean for variable 2
        step1 : float
            Increment of evaluations for variable 1
        step2 : float
            Increment of evaluations for variable 2
        levs : list
            Levels to be drawn on the contour plot.
            Default =  (.001, .01, .05, .1, .2)
        plot_dta : bool
            If True, makes a scatter plot of the data on
            top of the contour plot. Defaultis False.
        var1_name : str
            Name of variable 1 to be plotted on the x-axis
        var2_name : str
            Name of variable 2 to be plotted on the y-axis

        Notes
        -----
        The smaller the step size, the more accurate the intervals
        will be

        If the function returns optimization failed, consider narrowing
        the boundaries of the plot

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> two_rvs = np.random.standard_normal((20,2))
        >>> el_analysis = sm.emplike.DescStat(two_rvs)
        >>> contourp = el_analysis.mv_mean_contour(-2, 2, -2, 2, .1, .1)
        >>> contourp.show()
        """
        if self.endog.shape[1] != 2:
            raise ValueError('Data must contain exactly two variables')
        fig, ax = utils.create_mpl_ax()
        if var2_name is None:
            ax.set_ylabel('Variable 2')
        else:
            ax.set_ylabel(var2_name)
        if var1_name is None:
            ax.set_xlabel('Variable 1')
        else:
            ax.set_xlabel(var1_name)
        x = np.arange(mu1_low, mu1_upp, step1)
        y = np.arange(mu2_low, mu2_upp, step2)
        pairs = itertools.product(x, y)
        z = []
        for i in pairs:
            z.append(self.mv_test_mean(np.asarray(i))[0])
        X, Y = np.meshgrid(x, y)
        z = np.asarray(z)
        z = z.reshape(X.shape[1], Y.shape[0])
        ax.contour(x, y, z.T, levels=levs)
        if plot_dta:
            ax.plot(self.endog[:, 0], self.endog[:, 1], 'bo')
        return fig

    def test_corr(self, corr0, return_weights=0):
        """
        Returns -2 x log-likelihood ratio and  p-value for the
        correlation coefficient between 2 variables

        Parameters
        ----------
        corr0 : float
            Hypothesized value to be tested

        return_weights : bool
            If true, returns the weights that maximize
            the log-likelihood at the hypothesized value
        """
        nobs = self.nobs
        endog = self.endog
        if endog.shape[1] != 2:
            raise NotImplementedError('Correlation matrix not yet implemented')
        nuis0 = np.array([endog[:, 0].mean(), endog[:, 0].var(), endog[:, 1].mean(), endog[:, 1].var()])
        x0 = np.zeros(5)
        weights0 = np.array([1.0 / nobs] * int(nobs))
        args = (corr0, endog, nobs, x0, weights0)
        llr = optimize.fmin(self._opt_correl, nuis0, args=args, full_output=1, disp=0)[1]
        p_val = chi2.sf(llr, 1)
        if return_weights:
            return (llr, p_val, self.new_weights.T)
        return (llr, p_val)

    def ci_corr(self, sig=0.05, upper_bound=None, lower_bound=None):
        """
        Returns the confidence intervals for the correlation coefficient

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value the upper confidence limit can be.
            Default is  99% confidence limit assuming normality.

        lower_bound : float
            Minimum value the lower confidence limit can be.
            Default is 99% confidence limit assuming normality.

        Returns
        -------
        interval : tuple
            Confidence interval for the correlation
        """
        endog = self.endog
        nobs = self.nobs
        self.r0 = chi2.ppf(1 - sig, 1)
        point_est = np.corrcoef(endog[:, 0], endog[:, 1])[0, 1]
        if upper_bound is None:
            upper_bound = min(0.999, point_est + 2.5 * ((1.0 - point_est ** 2.0) / (nobs - 2.0)) ** 0.5)
        if lower_bound is None:
            lower_bound = max(-0.999, point_est - 2.5 * np.sqrt((1.0 - point_est ** 2.0) / (nobs - 2.0)))
        llim = optimize.brenth(self._ci_limits_corr, lower_bound, point_est)
        ulim = optimize.brenth(self._ci_limits_corr, point_est, upper_bound)
        return (llim, ulim)
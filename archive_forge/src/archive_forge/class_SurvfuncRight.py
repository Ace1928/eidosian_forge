import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
class SurvfuncRight:
    """
    Estimation and inference for a survival function.

    The survival function S(t) = P(T > t) is the probability that an
    event time T is greater than t.

    This class currently only supports right censoring.

    Parameters
    ----------
    time : array_like
        An array of times (censoring times or event times)
    status : array_like
        Status at the event time, status==1 is the 'event'
        (e.g. death, failure), meaning that the event
        occurs at the given value in `time`; status==0
        indicates that censoring has occurred, meaning that
        the event occurs after the given value in `time`.
    entry : array_like, optional An array of entry times for handling
        left truncation (the subject is not in the risk set on or
        before the entry time)
    title : str
        Optional title used for plots and summary output.
    freq_weights : array_like
        Optional frequency weights
    exog : array_like
        Optional, if present used to account for violation of
        independent censoring.
    bw_factor : float
        Band-width multiplier for kernel-based estimation.  Only used
        if exog is provided.

    Attributes
    ----------
    surv_prob : array_like
        The estimated value of the survivor function at each time
        point in `surv_times`.
    surv_prob_se : array_like
        The standard errors for the values in `surv_prob`.  Not available
        if exog is provided.
    surv_times : array_like
        The points where the survival function changes.
    n_risk : array_like
        The number of subjects at risk just before each time value in
        `surv_times`.  Not available if exog is provided.
    n_events : array_like
        The number of events (e.g. deaths) that occur at each point
        in `surv_times`.  Not available if exog is provided.

    Notes
    -----
    If exog is None, the standard Kaplan-Meier estimator is used.  If
    exog is not None, a local estimate of the marginal survival
    function around each point is constructed, and these are then
    averaged.  This procedure gives an estimate of the marginal
    survival function that accounts for dependent censoring as long as
    the censoring becomes independent when conditioning on the
    covariates in exog.  See Zeng et al. (2004) for details.

    References
    ----------
    D. Zeng (2004).  Estimating marginal survival function by
    adjusting for dependent censoring using many covariates.  Annals
    of Statistics 32:4.
    https://arxiv.org/pdf/math/0409180.pdf
    """

    def __init__(self, time, status, entry=None, title=None, freq_weights=None, exog=None, bw_factor=1.0):
        _checkargs(time, status, entry, freq_weights, exog)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)
        if entry is not None:
            entry = self.entry = np.asarray(entry)
        if exog is not None:
            if entry is not None:
                raise ValueError('exog and entry cannot both be present')
            from ._kernel_estimates import _kernel_survfunc
            exog = self.exog = np.asarray(exog)
            nobs = exog.shape[0]
            kw = nobs ** (-1 / 3.0) * bw_factor
            kfunc = lambda x: np.exp(-x ** 2 / kw ** 2).sum(1)
            x = _kernel_survfunc(time, status, exog, kfunc, freq_weights)
            self.surv_prob = x[0]
            self.surv_times = x[1]
            return
        x = _calc_survfunc_right(time, status, weights=freq_weights, entry=entry)
        self.surv_prob = x[0]
        self.surv_prob_se = x[1]
        self.surv_times = x[2]
        self.n_risk = x[4]
        self.n_events = x[5]
        self.title = '' if not title else title

    def plot(self, ax=None):
        """
        Plot the survival function.

        Examples
        --------
        Change the line color:

        >>> import statsmodels.api as sm
        >>> data = sm.datasets.get_rdataset("flchain", "survival").data
        >>> df = data.loc[data.sex == "F", :]
        >>> sf = sm.SurvfuncRight(df["futime"], df["death"])
        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> li = ax.get_lines()
        >>> li[0].set_color('purple')
        >>> li[1].set_color('purple')

        Do not show the censoring points:

        >>> fig = sf.plot()
        >>> ax = fig.get_axes()[0]
        >>> li = ax.get_lines()
        >>> li[1].set_visible(False)
        """
        return plot_survfunc(self, ax)

    def quantile(self, p):
        """
        Estimated quantile of a survival distribution.

        Parameters
        ----------
        p : float
            The probability point at which the quantile
            is determined.

        Returns the estimated quantile.
        """
        ii = np.flatnonzero(self.surv_prob < 1 - p)
        if len(ii) == 0:
            return np.nan
        return self.surv_times[ii[0]]

    def quantile_ci(self, p, alpha=0.05, method='cloglog'):
        """
        Returns a confidence interval for a survival quantile.

        Parameters
        ----------
        p : float
            The probability point for which a confidence interval is
            determined.
        alpha : float
            The confidence interval has nominal coverage probability
            1 - `alpha`.
        method : str
            Function to use for g-transformation, must be ...

        Returns
        -------
        lb : float
            The lower confidence limit.
        ub : float
            The upper confidence limit.

        Notes
        -----
        The confidence interval is obtained by inverting Z-tests.  The
        limits of the confidence interval will always be observed
        event times.

        References
        ----------
        The method is based on the approach used in SAS, documented here:

          http://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/viewer.htm#statug_lifetest_details03.htm
        """
        tr = norm.ppf(1 - alpha / 2)
        method = method.lower()
        if method == 'cloglog':
            g = lambda x: np.log(-np.log(x))
            gprime = lambda x: -1 / (x * np.log(x))
        elif method == 'linear':
            g = lambda x: x
            gprime = lambda x: 1
        elif method == 'log':
            g = np.log
            gprime = lambda x: 1 / x
        elif method == 'logit':
            g = lambda x: np.log(x / (1 - x))
            gprime = lambda x: 1 / (x * (1 - x))
        elif method == 'asinsqrt':
            g = lambda x: np.arcsin(np.sqrt(x))
            gprime = lambda x: 1 / (2 * np.sqrt(x) * np.sqrt(1 - x))
        else:
            raise ValueError('unknown method')
        r = g(self.surv_prob) - g(1 - p)
        r /= gprime(self.surv_prob) * self.surv_prob_se
        ii = np.flatnonzero(np.abs(r) <= tr)
        if len(ii) == 0:
            return (np.nan, np.nan)
        lb = self.surv_times[ii[0]]
        if ii[-1] == len(self.surv_times) - 1:
            ub = np.inf
        else:
            ub = self.surv_times[ii[-1] + 1]
        return (lb, ub)

    def summary(self):
        """
        Return a summary of the estimated survival function.

        The summary is a dataframe containing the unique event times,
        estimated survival function values, and related quantities.
        """
        df = pd.DataFrame(index=self.surv_times)
        df.index.name = 'Time'
        df['Surv prob'] = self.surv_prob
        df['Surv prob SE'] = self.surv_prob_se
        df['num at risk'] = self.n_risk
        df['num events'] = self.n_events
        return df

    def simultaneous_cb(self, alpha=0.05, method='hw', transform='log'):
        """
        Returns a simultaneous confidence band for the survival function.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the desired simultaneous coverage
            probability for the confidence region.  Currently alpha
            must be set to 0.05, giving 95% simultaneous intervals.
        method : str
            The method used to produce the simultaneous confidence
            band.  Only the Hall-Wellner (hw) method is currently
            implemented.
        transform : str
            The used to produce the interval (note that the returned
            interval is on the survival probability scale regardless
            of which transform is used).  Only `log` and `arcsin` are
            implemented.

        Returns
        -------
        lcb : array_like
            The lower confidence limits corresponding to the points
            in `surv_times`.
        ucb : array_like
            The upper confidence limits corresponding to the points
            in `surv_times`.
        """
        method = method.lower()
        if method != 'hw':
            msg = 'only the Hall-Wellner (hw) method is implemented'
            raise ValueError(msg)
        if alpha != 0.05:
            raise ValueError('alpha must be set to 0.05')
        transform = transform.lower()
        s2 = self.surv_prob_se ** 2 / self.surv_prob ** 2
        nn = self.n_risk
        if transform == 'log':
            denom = np.sqrt(nn) * np.log(self.surv_prob)
            theta = 1.3581 * (1 + nn * s2) / denom
            theta = np.exp(theta)
            lcb = self.surv_prob ** (1 / theta)
            ucb = self.surv_prob ** theta
        elif transform == 'arcsin':
            k = 1.3581
            k *= (1 + nn * s2) / (2 * np.sqrt(nn))
            k *= np.sqrt(self.surv_prob / (1 - self.surv_prob))
            f = np.arcsin(np.sqrt(self.surv_prob))
            v = np.clip(f - k, 0, np.inf)
            lcb = np.sin(v) ** 2
            v = np.clip(f + k, -np.inf, np.pi / 2)
            ucb = np.sin(v) ** 2
        else:
            raise ValueError('Unknown transform')
        return (lcb, ucb)
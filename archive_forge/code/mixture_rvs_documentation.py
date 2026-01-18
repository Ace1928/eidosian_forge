import numpy as np

        cdf of a mixture of distributions.

        Parameters
        ----------
        x : array_like
            Array containing locations where the CDF should be evaluated
        prob : array_like
            Probability of sampling from each distribution in dist
        size : int
            The length of the returned sample.
        dist : array_like
            An iterable of distributions objects from scipy.stats.
        kwargs : tuple of dicts, optional
            A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and
            args to be passed to the respective distribution in dist.  If not
            provided, the distribution defaults are used.

        Examples
        --------
        Say we want 5000 random variables from mixture of normals with two
        distributions norm(-1,.5) and norm(1,.5) and we want to sample from the
        first with probability .75 and the second with probability .25.

        >>> import numpy as np
        >>> from scipy import stats
        >>> from statsmodels.distributions.mixture_rvs import MixtureDistribution
        >>> x = np.arange(-4.0, 4.0, 0.01)
        >>> prob = [.75,.25]
        >>> mixture = MixtureDistribution()
        >>> Y = mixture.pdf(x, prob, dist=[stats.norm, stats.norm],
        ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
        
import numpy as np
from scipy import stats
class ClippedContinuous:
    """clipped continuous distribution with a masspoint at clip_lower


    Notes
    -----
    first version, to try out possible designs
    insufficient checks for valid arguments and not clear
    whether it works for distributions that have compact support

    clip_lower is fixed and independent of the distribution parameters.
    The clip_lower point in the pdf has to be interpreted as a mass point,
    i.e. different treatment in integration and expect function, which means
    none of the generic methods for this can be used.

    maybe this will be better designed as a mixture between a degenerate or
    discrete and a continuous distribution

    Warning: uses equality to check for clip_lower values in function
    arguments, since these are floating points, the comparison might fail
    if clip_lower values are not exactly equal.
    We could add a check whether the values are in a small neighborhood, but
    it would be expensive (need to search and check all values).

    """

    def __init__(self, base_dist, clip_lower):
        self.base_dist = base_dist
        self.clip_lower = clip_lower

    def _get_clip_lower(self, kwds):
        """helper method to get clip_lower from kwds or attribute

        """
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            clip_lower = kwds.pop('clip_lower')
        return (clip_lower, kwds)

    def rvs(self, *args, **kwds):
        clip_lower, kwds = self._get_clip_lower(kwds)
        rvs_ = self.base_dist.rvs(*args, **kwds)
        rvs_[rvs_ < clip_lower] = clip_lower
        return rvs_

    def pdf(self, x, *args, **kwds):
        x = np.atleast_1d(x)
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            clip_lower = kwds.pop('clip_lower')
        pdf_raw = np.atleast_1d(self.base_dist.pdf(x, *args, **kwds))
        clip_mask = x == self.clip_lower
        if np.any(clip_mask):
            clip_prob = self.base_dist.cdf(clip_lower, *args, **kwds)
            pdf_raw[clip_mask] = clip_prob
        pdf_raw[x < clip_lower] = 0
        return pdf_raw

    def cdf(self, x, *args, **kwds):
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            clip_lower = kwds.pop('clip_lower')
        cdf_raw = self.base_dist.cdf(x, *args, **kwds)
        cdf_raw[x < clip_lower] = 0
        return cdf_raw

    def sf(self, x, *args, **kwds):
        if 'clip_lower' not in kwds:
            clip_lower = self.clip_lower
        else:
            clip_lower = kwds.pop('clip_lower')
        sf_raw = self.base_dist.sf(x, *args, **kwds)
        sf_raw[x <= clip_lower] = 1
        return sf_raw

    def ppf(self, x, *args, **kwds):
        raise NotImplementedError

    def plot(self, x, *args, **kwds):
        clip_lower, kwds = self._get_clip_lower(kwds)
        mass = self.pdf(clip_lower, *args, **kwds)
        xr = np.concatenate(([clip_lower + 1e-06], x[x > clip_lower]))
        import matplotlib.pyplot as plt
        plt.xlim(clip_lower - 0.1, x.max())
        xpdf = self.pdf(x, *args, **kwds)
        plt.ylim(0, max(mass, xpdf.max()) * 1.1)
        plt.plot(xr, self.pdf(xr, *args, **kwds))
        plt.stem([clip_lower], [mass], linefmt='b-', markerfmt='bo', basefmt='r-')
        return
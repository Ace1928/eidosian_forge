import numpy as np
import param
from ..core import Dataset, Dimension, NdOverlay
from ..core.operation import Operation
from ..core.util import cartesian_product, isfinite
from ..element import Area, Bivariate, Contours, Curve, Distribution, Image, Polygons
from .element import contours
class bivariate_kde(Operation):
    """
    Computes a 2D kernel density estimate (KDE) of the first two
    dimensions in the input data. Kernel density estimation is a
    non-parametric way to estimate the probability density function of
    a random variable.

    The KDE works by placing 2D Gaussian kernel at each sample with
    the supplied bandwidth. These kernels are then summed to produce
    the density estimate. By default a good bandwidth is determined
    using the bw_method but it may be overridden by an explicit value.
    """
    contours = param.Boolean(default=True, doc='\n        Whether to compute contours from the KDE, determines whether to\n        return an Image or Contours/Polygons.')
    bw_method = param.ObjectSelector(default='scott', objects=['scott', 'silverman'], doc='\n        Method of automatically determining KDE bandwidth')
    bandwidth = param.Number(default=None, doc='\n        Allows supplying explicit bandwidth value rather than relying\n        on scott or silverman method.')
    cut = param.Number(default=3, doc='\n        Draw the estimate to cut * bw from the extreme data points.')
    filled = param.Boolean(default=False, doc='\n        Controls whether to return filled or unfilled contours.')
    levels = param.ClassSelector(default=10, class_=(list, int), doc='\n        A list of scalar values used to specify the contour levels.')
    n_samples = param.Integer(default=100, doc='\n        Number of samples to compute the KDE over.')
    x_range = param.NumericTuple(default=None, length=2, doc='\n       The x_range as a tuple of min and max x-value. Auto-ranges\n       if set to None.')
    y_range = param.NumericTuple(default=None, length=2, doc='\n       The x_range as a tuple of min and max y-value. Auto-ranges\n       if set to None.')
    _per_element = True

    def _process(self, element, key=None):
        try:
            from scipy import stats
        except ImportError:
            raise ImportError(f'{type(self).__name__} operation requires SciPy to be installed.') from None
        if len(element.dimensions()) < 2:
            raise ValueError('bivariate_kde can only be computed on elements declaring at least two dimensions.')
        xdim, ydim = element.dimensions()[:2]
        params = {}
        if isinstance(element, Bivariate):
            if element.group != type(element).__name__:
                params['group'] = element.group
            params['label'] = element.label
            vdim = element.vdims[0]
        else:
            vdim = 'Density'
        data = element.array([0, 1]).T
        xmin, xmax = self.p.x_range or element.range(0)
        ymin, ymax = self.p.y_range or element.range(1)
        if any((not isfinite(v) for v in (xmin, xmax))):
            xmin, xmax = (-0.5, 0.5)
        elif xmin == xmax:
            xmin, xmax = (xmin - 0.5, xmax + 0.5)
        if any((not isfinite(v) for v in (ymin, ymax))):
            ymin, ymax = (-0.5, 0.5)
        elif ymin == ymax:
            ymin, ymax = (ymin - 0.5, ymax + 0.5)
        data = data[:, isfinite(data).min(axis=0)] if data.shape[1] > 1 else np.empty((2, 0))
        if data.shape[1] > 1:
            kde = stats.gaussian_kde(data)
            if self.p.bandwidth:
                kde.set_bandwidth(self.p.bandwidth)
            bw = kde.scotts_factor() * data.std(ddof=1)
            if self.p.x_range:
                xs = np.linspace(xmin, xmax, self.p.n_samples)
            else:
                xs = _kde_support((xmin, xmax), bw, self.p.n_samples, self.p.cut, xdim.range)
            if self.p.y_range:
                ys = np.linspace(ymin, ymax, self.p.n_samples)
            else:
                ys = _kde_support((ymin, ymax), bw, self.p.n_samples, self.p.cut, ydim.range)
            xx, yy = cartesian_product([xs, ys], False)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kde(positions).T, xx.shape)
        elif self.p.contours:
            eltype = Polygons if self.p.filled else Contours
            return eltype([], kdims=[xdim, ydim], vdims=[vdim])
        else:
            xs = np.linspace(xmin, xmax, self.p.n_samples)
            ys = np.linspace(ymin, ymax, self.p.n_samples)
            f = np.zeros((self.p.n_samples, self.p.n_samples))
        img = Image((xs, ys, f.T), kdims=element.dimensions()[:2], vdims=[vdim], **params)
        if self.p.contours:
            cntr = contours(img, filled=self.p.filled, levels=self.p.levels)
            return cntr.clone(cntr.data[1:], **params)
        return img
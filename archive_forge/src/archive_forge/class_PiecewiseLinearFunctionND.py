from collections.abc import Sized
import logging
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.variable import variable, variable_dict, variable_tuple
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.expression import expression, expression_tuple
import pyomo.core.kernel.piecewise_library.util
class PiecewiseLinearFunctionND(object):
    """A multi-variate piecewise linear function

    Multi-varite piecewise linear functions are defined by a
    triangulation over a finite domain and a list of
    function values associated with the points of the
    triangulation.  The function value between points in the
    triangulation is implied through linear interpolation.

    Args:
        tri (scipy.spatial.Delaunay): A triangulation over
            the discretized variable domain. Can be
            generated using a list of variables using the
            utility function :func:`util.generate_delaunay`.
            Required attributes:

              - points: An (npoints, D) shaped array listing
                the D-dimensional coordinates of the
                discretization points.
              - simplices: An (nsimplices, D+1) shaped array
                of integers specifying the D+1 indices of
                the points vector that define each simplex
                of the triangulation.
        values (numpy.array): An (npoints,) shaped array of
            the values of the piecewise function at each of
            coordinates in the triangulation points array.
    """
    __slots__ = ('_tri', '_values')

    def __init__(self, tri, values, validate=True, **kwds):
        assert pyomo.core.kernel.piecewise_library.util.numpy_available
        assert pyomo.core.kernel.piecewise_library.util.scipy_available
        assert isinstance(tri, pyomo.core.kernel.piecewise_library.util.scipy.spatial.Delaunay)
        assert isinstance(values, pyomo.core.kernel.piecewise_library.util.numpy.ndarray)
        npoints, ndim = tri.points.shape
        nsimplices, _ = tri.simplices.shape
        assert tri.simplices.shape[1] == ndim + 1
        assert nsimplices > 0
        assert npoints > 0
        assert ndim > 0
        self._tri = tri
        self._values = values

    def __getstate__(self):
        """Required for older versions of the pickle
        protocol since this class uses __slots__"""
        return {key: getattr(self, key) for key in self.__slots__}

    def __setstate__(self, state):
        """Required for older versions of the pickle
        protocol since this class uses __slots__"""
        for key in state:
            setattr(self, key, state[key])

    @property
    def triangulation(self):
        """The triangulation over the domain of this function"""
        return self._tri

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._values

    def __call__(self, x):
        """
        Evaluates the piecewise linear function using
        interpolation. This method supports vectorized
        function calls as the interpolation process can be
        expensive for high dimensional data.

        For the case when a single point is provided, the
        argument x should be a (D,) shaped numpy array or
        list, where D is the dimension of points in the
        triangulation.

        For the vectorized case, the argument x should be
        a (n,D)-shaped numpy array.
        """
        assert isinstance(x, Sized)
        if isinstance(x, pyomo.core.kernel.piecewise_library.util.numpy.ndarray):
            if x.shape != self._tri.points.shape[1:]:
                multi = True
                assert x.shape[1:] == self._tri.points[0].shape, '%s[1] != %s' % (x.shape, self._tri.points[0].shape)
            else:
                multi = False
        else:
            multi = False
        _, ndim = self._tri.points.shape
        i = self._tri.find_simplex(x)
        if multi:
            Tinv = self._tri.transform[i, :ndim]
            r = self._tri.transform[i, ndim]
            b = pyomo.core.kernel.piecewise_library.util.numpy.einsum('ijk,ik->ij', Tinv, x - r)
            b = pyomo.core.kernel.piecewise_library.util.numpy.c_[b, 1 - b.sum(axis=1)]
            s = self._tri.simplices[i]
            return (b * self._values[s]).sum(axis=1)
        else:
            b = self._tri.transform[i, :ndim, :ndim].dot(x - self._tri.transform[i, ndim, :])
            s = self._tri.simplices[i]
            val = b.dot(self._values[s[:ndim]])
            val += (1 - b.sum()) * self._values[s[ndim]]
            return val
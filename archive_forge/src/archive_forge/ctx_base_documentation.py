from operator import gt, lt
from .libmp.backend import xrange
from .functions.functions import SpecialFunctions
from .functions.rszeta import RSCache
from .calculus.quadrature import QuadratureMethods
from .calculus.inverselaplace import LaplaceTransformInversionMethods
from .calculus.calculus import CalculusMethods
from .calculus.optimization import OptimizationMethods
from .calculus.odes import ODEMethods
from .matrices.matrices import MatrixMethods
from .matrices.calculus import MatrixCalculusMethods
from .matrices.linalg import LinearAlgebraMethods
from .matrices.eigen import Eigen
from .identification import IdentificationMethods
from .visualization import VisualizationMethods
from . import libmp

        Return a wrapped copy of *f* that caches computed values, i.e.
        a memoized copy of *f*. Values are only reused if the cached precision
        is equal to or higher than the working precision::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = memoize(maxcalls(sin, 1))
            >>> f(2)
            0.909297426825682
            >>> f(2)
            0.909297426825682
            >>> mp.dps = 25
            >>> f(2) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 1 times

        
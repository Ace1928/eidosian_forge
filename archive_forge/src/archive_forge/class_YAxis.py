import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
class YAxis(Axis):
    _yaxis = yaxis
    _yaxisgrob = grid.yaxisGrob

    @classmethod
    def yaxis(cls, **kwargs):
        """ Constructor (uses the R function grid::yaxis())"""
        res = cls._yaxis(**kwargs)
        return cls(res)

    @classmethod
    def yaxisgrob(cls, **kwargs):
        """ Constructor (uses the R function grid::yaxisgrob())"""
        res = cls._yaxisgrob(**kwargs)
        return cls(res)
import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
class Grob(BaseGrid):
    """ Graphical object """
    _r_constructor = grid_env['grob']
    _draw = grid_env['grid.draw']

    def draw(self, recording=True):
        """ Draw a graphical object (calling the R function
        grid::grid.raw())"""
        self._draw(self, recording=recording)
import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
class Options(robjects.ListVector):

    def __repr__(self):
        s = '<instance of %s : %i>' % (type(self), id(self))
        return s
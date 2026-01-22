import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
class Stat(GBaseObject):
    """ A "statistical" processing of the data in order
    to make a plot, or a plot element.

    This is an abstract class; material classes are called
    Stat* (e.g., StatAbline, StatBin, etc...). """
    pass
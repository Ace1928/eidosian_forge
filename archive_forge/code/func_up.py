import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
@classmethod
def up(cls, n, recording=True):
    """ Go up n viewports """
    cls._downviewport(n, recording=recording)
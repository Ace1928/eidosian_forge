import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
class ElementBlank(Theme):
    _constructor = ggplot2.element_blank

    @classmethod
    def new(cls):
        res = cls(cls._constructor())
        return res
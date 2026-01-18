from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def transmute(self, *args, **kwargs):
    """Call the R function `dplyr::transmute()`."""
    res = dplyr.transmute(self, *args, **kwargs)
    return res
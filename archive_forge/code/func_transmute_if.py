from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def transmute_if(self, *args, **kwargs):
    """Call the R function `dplyr::transmute_if()`."""
    res = dplyr.transmute_if(self, *args, **kwargs)
    return res
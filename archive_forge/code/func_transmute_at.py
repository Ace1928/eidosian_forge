from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def transmute_at(self, *args, **kwargs):
    """Call the R function `dplyr::transmute_at()`."""
    res = dplyr.transmute_at(self, *args, **kwargs)
    return res
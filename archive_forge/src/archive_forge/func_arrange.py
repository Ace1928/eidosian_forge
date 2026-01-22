from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def arrange(self, *args, _by_group=False):
    """Call the R function `dplyr::arrange()`."""
    res = dplyr.arrange(self, *args, **{'.by_group': _by_group})
    return res
from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def _wrap(rfunc, cls):

    def func(dataf, *args, **kwargs):
        res = rfunc(dataf, *args, **kwargs)
        if cls is None:
            return type(dataf)(res)
        else:
            return cls(res)
    return func
from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def _fix_call(func):

    def inner(self, *args, **kwargs):
        if len(args) > 0:
            if len(kwargs) > 0:
                res = func(self, *args, **kwargs)
            else:
                res = func(self, *args)
        elif len(kwargs) > 0:
            res = func(self, **kwargs)
        else:
            res = func(self)
        return res
    return inner
from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def transform_data(self, x, initialize=False):
    tm = self.transform_data_method
    if tm is None:
        return x
    if initialize is True:
        if tm == 'domain':
            self.domain_low = x.min(0)
            self.domain_upp = x.max(0)
        elif isinstance(tm, tuple):
            self.domain_low = tm[0]
            self.domain_upp = tm[1]
            self.transform_data_method = 'domain'
        else:
            raise ValueError("transform should be None, 'domain' or a tuple")
        self.domain_diff = self.domain_upp - self.domain_low
    if self.transform_data_method == 'domain':
        x = (x - self.domain_low) / self.domain_diff
        return x
    else:
        raise ValueError('incorrect transform_data_method')
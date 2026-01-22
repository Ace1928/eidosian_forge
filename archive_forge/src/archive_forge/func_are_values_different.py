import hashlib
import os
import re
import sys
from collections import OrderedDict
from optparse import Option, OptionParser
import numpy as np
import nibabel as nib
import nibabel.cmdline.utils
def are_values_different(*values):
    """Generically compare values, return True if different

    Note that comparison is targeting reporting of comparison of the headers
    so has following specifics:
    - even a difference in data types is considered a difference, i.e. 1 != 1.0
    - nans are considered to be the "same", although generally nan != nan
    """
    value0 = values[0]
    if isinstance(value0, np.ndarray):
        try:
            value0_nans = np.asanyarray(np.isnan(value0))
            value0_nonnans = np.asanyarray(np.logical_not(value0_nans))
            if not np.any(value0_nans):
                value0_nans = None
        except TypeError as exc:
            str_exc = str(exc)
            if 'not supported' in str_exc or 'not implemented' in str_exc:
                value0_nans = None
            else:
                raise
    for value in values[1:]:
        if type(value0) != type(value):
            return True
        elif isinstance(value0, np.ndarray):
            if value0.dtype.type != value.dtype.type or value0.shape != value.shape:
                return True
            if value0_nans is not None:
                value_nans = np.isnan(value)
                if np.any(value0_nans != value_nans):
                    return True
                if np.any(value0[value0_nonnans] != value[value0_nonnans]):
                    return True
            elif np.any(value0 != value):
                return True
        elif value0 is np.nan:
            if value is not np.nan:
                return True
        elif value0 != value:
            return True
    return False
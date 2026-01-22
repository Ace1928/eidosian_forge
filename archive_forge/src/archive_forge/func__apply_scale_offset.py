import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
def _apply_scale_offset(self, data, scale, offset):
    if scale != 1:
        if offset == 0:
            return data * scale
        return data * scale + offset
    if offset != 0:
        return data + offset
    return data
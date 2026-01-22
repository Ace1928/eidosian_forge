import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class Overlap(nam.Overlap):
    """Calculates various overlap measures between two maps.

    .. deprecated:: 0.10.0
       Use :py:class:`nipype.algorithms.metrics.Overlap` instead.
    """

    def __init__(self, **inputs):
        super(nam.Overlap, self).__init__(**inputs)
        warnings.warn('This interface has been deprecated since 0.10.0, please use nipype.algorithms.metrics.Overlap', DeprecationWarning)
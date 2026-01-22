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
def _gen_output_filename(self):
    if not isdefined(self.inputs.out_file):
        _, base, ext = split_filename(self.inputs.in_file)
        out_file = os.path.abspath('%s_SNR%03.2f%s' % (base, self.inputs.snr, ext))
    else:
        out_file = self.inputs.out_file
    return out_file
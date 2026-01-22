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
def _gen_output_file_name(self):
    _, base, ext = split_filename(self.inputs.in_file)
    if self.inputs.mode == 'decompress' and ext[-3:].lower() == '.gz':
        ext = ext[:-3]
    elif self.inputs.mode == 'compress':
        ext = f'{ext}.gz'
    return os.path.abspath(base + ext)
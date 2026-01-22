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
class GzipInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='file to (de)compress')
    mode = traits.Enum('compress', 'decompress', usedefault=True, desc='compress or decompress')
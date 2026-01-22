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
class AddCSVColumnInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='Input comma-separated value (CSV) files')
    out_file = File('extra_heading.csv', usedefault=True, desc='Output filename for merged CSV file')
    extra_column_heading = traits.Str(desc='New heading to add for the added field.')
    extra_field = traits.Str(desc='New field to add to each row. This is useful for saving the        group or subject ID in the file.')
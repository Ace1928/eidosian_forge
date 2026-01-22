import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FNIRTOutputSpec(TraitedSpec):
    fieldcoeff_file = File(exists=True, desc='file with field coefficients')
    warped_file = File(exists=True, desc='warped image')
    field_file = File(desc='file with warp field')
    jacobian_file = File(desc='file containing Jacobian of the field')
    modulatedref_file = File(desc='file containing intensity modulated --ref')
    out_intensitymap_file = traits.List(File, minlen=2, maxlen=2, desc='files containing info pertaining to intensity mapping')
    log_file = File(desc='Name of log-file')
import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
def _get_encfilename(self):
    out_file = os.path.join(os.getcwd(), '%s_encfile.txt' % split_filename(self.inputs.in_file)[1])
    return out_file
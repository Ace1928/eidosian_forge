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
class AddNoiseInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='input image that will be corrupted with noise')
    in_mask = File(exists=True, desc='input mask, voxels outside this mask will be considered background')
    snr = traits.Float(10.0, desc='desired output SNR in dB', usedefault=True)
    dist = traits.Enum('normal', 'rician', usedefault=True, mandatory=True, desc='desired noise distribution')
    bg_dist = traits.Enum('normal', 'rayleigh', usedefault=True, mandatory=True, desc='desired noise distribution, currently only normal is implemented')
    out_file = File(desc='desired output filename')
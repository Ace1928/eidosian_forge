import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
class TSNRInputSpec(BaseInterfaceInputSpec):
    in_file = InputMultiPath(File(exists=True), mandatory=True, desc='realigned 4D file or a list of 3D files')
    regress_poly = traits.Range(low=1, desc='Remove polynomials')
    tsnr_file = File('tsnr.nii.gz', usedefault=True, hash_files=False, desc='output tSNR file')
    mean_file = File('mean.nii.gz', usedefault=True, hash_files=False, desc='output mean file')
    stddev_file = File('stdev.nii.gz', usedefault=True, hash_files=False, desc='output tSNR file')
    detrended_file = File('detrend.nii.gz', usedefault=True, hash_files=False, desc='input file after detrending')
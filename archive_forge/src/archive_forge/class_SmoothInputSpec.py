import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class SmoothInputSpec(FSTraitedSpec):
    in_file = File(exists=True, desc='source volume', argstr='--i %s', mandatory=True)
    reg_file = File(desc='registers volume to surface anatomical ', argstr='--reg %s', mandatory=True, exists=True)
    smoothed_file = File(desc='output volume', argstr='--o %s', genfile=True)
    proj_frac_avg = traits.Tuple(traits.Float, traits.Float, traits.Float, xor=['proj_frac'], desc='average a long normal min max delta', argstr='--projfrac-avg %.2f %.2f %.2f')
    proj_frac = traits.Float(desc='project frac of thickness a long surface normal', xor=['proj_frac_avg'], argstr='--projfrac %s')
    surface_fwhm = traits.Range(low=0.0, requires=['reg_file'], mandatory=True, xor=['num_iters'], desc='surface FWHM in mm', argstr='--fwhm %f')
    num_iters = traits.Range(low=1, xor=['surface_fwhm'], mandatory=True, argstr='--niters %d', desc='number of iterations instead of fwhm')
    vol_fwhm = traits.Range(low=0.0, argstr='--vol-fwhm %f', desc='volume smoothing outside of surface')
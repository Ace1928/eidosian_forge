import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class CoregisterInputSpec(SPMCommandInputSpec):
    target = ImageFileSPM(exists=True, mandatory=True, field='ref', desc='reference file to register to', copyfile=False)
    source = InputMultiPath(ImageFileSPM(exists=True), field='source', desc='file to register to target', copyfile=True, mandatory=True)
    jobtype = traits.Enum('estwrite', 'estimate', 'write', desc='one of: estimate, write, estwrite', usedefault=True)
    apply_to_files = InputMultiPath(File(exists=True), field='other', desc='files to apply transformation to', copyfile=True)
    cost_function = traits.Enum('mi', 'nmi', 'ecc', 'ncc', field='eoptions.cost_fun', desc="cost function, one of:\n                    'mi' - Mutual Information,\n                    'nmi' - Normalised Mutual Information,\n                    'ecc' - Entropy Correlation Coefficient,\n                    'ncc' - Normalised Cross Correlation")
    fwhm = traits.List(traits.Float(), minlen=2, maxlen=2, field='eoptions.fwhm', desc='gaussian smoothing kernel width (mm)')
    separation = traits.List(traits.Float(), field='eoptions.sep', desc='sampling separation in mm')
    tolerance = traits.List(traits.Float(), field='eoptions.tol', desc='acceptable tolerance for each of 12 params')
    write_interp = traits.Range(low=0, high=7, field='roptions.interp', desc='degree of b-spline used for interpolation')
    write_wrap = traits.List(traits.Int(), minlen=3, maxlen=3, field='roptions.wrap', desc='Check if interpolation should wrap in [x,y,z]')
    write_mask = traits.Bool(field='roptions.mask', desc='True/False mask output image')
    out_prefix = traits.String('r', field='roptions.prefix', usedefault=True, desc='coregistered output prefix')
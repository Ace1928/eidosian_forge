import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class RealignInputSpec(SPMCommandInputSpec):
    in_files = InputMultiPath(traits.Either(ImageFileSPM(exists=True), traits.List(ImageFileSPM(exists=True))), field='data', mandatory=True, copyfile=True, desc='list of filenames to realign')
    jobtype = traits.Enum('estwrite', 'estimate', 'write', desc='one of: estimate, write, estwrite', usedefault=True)
    quality = traits.Range(low=0.0, high=1.0, field='eoptions.quality', desc='0.1 = fast, 1.0 = precise')
    fwhm = traits.Range(low=0.0, field='eoptions.fwhm', desc='gaussian smoothing kernel width')
    separation = traits.Range(low=0.0, field='eoptions.sep', desc='sampling separation in mm')
    register_to_mean = traits.Bool(field='eoptions.rtm', desc='Indicate whether realignment is done to the mean image')
    weight_img = File(exists=True, field='eoptions.weight', desc='filename of weighting image')
    interp = traits.Range(low=0, high=7, field='eoptions.interp', desc='degree of b-spline used for interpolation')
    wrap = traits.List(traits.Int(), minlen=3, maxlen=3, field='eoptions.wrap', desc='Check if interpolation should wrap in [x,y,z]')
    write_which = traits.ListInt([2, 1], field='roptions.which', minlen=2, maxlen=2, usedefault=True, desc='determines which images to reslice')
    write_interp = traits.Range(low=0, high=7, field='roptions.interp', desc='degree of b-spline used for interpolation')
    write_wrap = traits.List(traits.Int(), minlen=3, maxlen=3, field='roptions.wrap', desc='Check if interpolation should wrap in [x,y,z]')
    write_mask = traits.Bool(field='roptions.mask', desc='True/False mask output image')
    out_prefix = traits.String('r', field='roptions.prefix', usedefault=True, desc='realigned output prefix')
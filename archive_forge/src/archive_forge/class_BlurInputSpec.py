import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BlurInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file_base = File(desc='output file base', argstr='%s', position=-1)
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    _xor_kernel = ('gaussian', 'rect')
    gaussian = traits.Bool(desc='Use a gaussian smoothing kernel (default).', argstr='-gaussian', xor=_xor_kernel)
    rect = traits.Bool(desc='Use a rect (box) smoothing kernel.', argstr='-rect', xor=_xor_kernel)
    gradient = traits.Bool(desc='Create the gradient magnitude volume as well.', argstr='-gradient')
    partial = traits.Bool(desc='Create the partial derivative and gradient magnitude volumes as well.', argstr='-partial')
    no_apodize = traits.Bool(desc='Do not apodize the data before blurring.', argstr='-no_apodize')
    _xor_main_options = ('fwhm', 'fwhm3d', 'standard_dev')
    fwhm = traits.Float(0, desc='Full-width-half-maximum of gaussian kernel. Default value: 0.', argstr='-fwhm %s', xor=_xor_main_options, mandatory=True)
    standard_dev = traits.Float(0, desc='Standard deviation of gaussian kernel. Default value: 0.', argstr='-standarddev %s', xor=_xor_main_options, mandatory=True)
    fwhm3d = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='-3dfwhm %s %s %s', desc='Full-width-half-maximum of gaussian kernel.Default value: -1.79769e+308 -1.79769e+308 -1.79769e+308.', xor=_xor_main_options, mandatory=True)
    dimensions = traits.Enum(3, 1, 2, desc='Number of dimensions to blur (either 1,2 or 3). Default value: 3.', argstr='-dimensions %s')
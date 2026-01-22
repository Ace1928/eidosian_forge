import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class VolSymmInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-3)
    trans_file = File(desc='output xfm trans file', genfile=True, argstr='%s', position=-2, name_source=['input_file'], hash_files=False, name_template='%s_vol_symm.xfm', keep_extension=False)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_vol_symm.mnc')
    input_grid_files = InputMultiPath(File, desc='input grid file(s)')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    fit_linear = traits.Bool(desc='Fit using a linear xfm.', argstr='-linear')
    fit_nonlinear = traits.Bool(desc='Fit using a non-linear xfm.', argstr='-nonlinear')
    nofit = traits.Bool(desc='Use the input transformation instead of generating one.', argstr='-nofit')
    config_file = File(desc='File containing the fitting configuration (nlpfit -help for info).', argstr='-config_file %s', exists=True)
    x = traits.Bool(desc='Flip volume in x-plane (default).', argstr='-x')
    y = traits.Bool(desc='Flip volume in y-plane.', argstr='-y')
    z = traits.Bool(desc='Flip volume in z-plane.', argstr='-z')
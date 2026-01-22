import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class XfmAvgInputSpec(CommandLineInputSpec):
    input_files = InputMultiPath(File(exists=True), desc='input file(s)', mandatory=True, sep=' ', argstr='%s', position=-2)
    input_grid_files = InputMultiPath(File, desc='input grid file(s)')
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1)
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    avg_linear = traits.Bool(desc='average the linear part [default].', argstr='-avg_linear')
    avg_nonlinear = traits.Bool(desc='average the non-linear part [default].', argstr='-avg_nonlinear')
    ignore_linear = traits.Bool(desc='opposite of -avg_linear.', argstr='-ignore_linear')
    ignore_nonlinear = traits.Bool(desc='opposite of -avg_nonlinear.', argstr='-ignore_nonline')
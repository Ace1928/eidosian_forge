import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class XfmConcatInputSpec(CommandLineInputSpec):
    input_files = InputMultiPath(File(exists=True), desc='input file(s)', mandatory=True, sep=' ', argstr='%s', position=-2)
    input_grid_files = InputMultiPath(File, desc='input grid file(s)')
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_files'], hash_files=False, name_template='%s_xfmconcat.xfm')
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
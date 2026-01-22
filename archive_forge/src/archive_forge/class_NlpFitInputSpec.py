import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class NlpFitInputSpec(CommandLineInputSpec):
    source = File(desc='source Minc file', exists=True, mandatory=True, argstr='%s', position=-3)
    target = File(desc='target Minc file', exists=True, mandatory=True, argstr='%s', position=-2)
    output_xfm = File(desc='output xfm file', genfile=True, argstr='%s', position=-1)
    input_grid_files = InputMultiPath(File, desc='input grid file(s)')
    config_file = File(desc='File containing the fitting configuration use.', argstr='-config_file %s', mandatory=True, exists=True)
    init_xfm = File(desc='Initial transformation (default identity).', argstr='-init_xfm %s', mandatory=True, exists=True)
    source_mask = File(desc='Source mask to use during fitting.', argstr='-source_mask %s', mandatory=True, exists=True)
    verbose = traits.Bool(desc='Print out log messages. Default: False.', argstr='-verbose')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
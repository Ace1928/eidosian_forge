import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class GenerateDirectionsInputSpec(CommandLineInputSpec):
    num_dirs = traits.Int(mandatory=True, argstr='%s', position=-2, desc='the number of directions to generate.')
    power = traits.Float(argstr='-power %s', desc='specify exponent to use for repulsion power law.')
    niter = traits.Int(argstr='-niter %s', desc='specify the maximum number of iterations to perform.')
    display_info = traits.Bool(argstr='-info', desc='Display information messages.')
    quiet_display = traits.Bool(argstr='-quiet', desc='do not display information messages or progress status.')
    display_debug = traits.Bool(argstr='-debug', desc='Display debugging messages.')
    out_file = File(name_source=['num_dirs'], name_template='directions_%d.txt', argstr='%s', hash_files=False, position=-1, desc='the text file to write the directions to, as [ az el ] pairs.')
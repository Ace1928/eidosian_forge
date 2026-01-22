import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ImageMathsInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=1)
    in_file2 = File(exists=True, argstr='%s', position=3)
    mask_file = File(exists=True, argstr='-mas %s', desc='use (following image>0) to mask current image')
    out_file = File(argstr='%s', position=-2, genfile=True, hash_files=False)
    op_string = traits.Str(argstr='%s', position=2, desc='string defining the operation, i. e. -add')
    suffix = traits.Str(desc='out_file suffix')
    out_data_type = traits.Enum('char', 'short', 'int', 'float', 'double', 'input', argstr='-odt %s', position=-1, desc='output datatype, one of (char, short, int, float, double, input)')
import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class ConvertInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file for converting', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_convert_output.mnc')
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    two = traits.Bool(desc='Create a MINC 2 output file.', argstr='-2')
    template = traits.Bool(desc='Create a template file. The dimensions, variables, andattributes of the input file are preserved but all data it set to zero.', argstr='-template')
    compression = traits.Enum(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, argstr='-compress %s', desc='Set the compression level, from 0 (disabled) to 9 (maximum).')
    chunk = traits.Range(low=0, desc='Set the target block size for chunking (0 default, >1 block size).', argstr='-chunk %d')
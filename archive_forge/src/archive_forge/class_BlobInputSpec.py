import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BlobInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file to blob', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_blob.mnc')
    trace = traits.Bool(desc='compute the trace (approximate growth and shrinkage) -- FAST', argstr='-trace')
    determinant = traits.Bool(desc='compute the determinant (exact growth and shrinkage) -- SLOW', argstr='-determinant')
    translation = traits.Bool(desc='compute translation (structure displacement)', argstr='-translation')
    magnitude = traits.Bool(desc='compute the magnitude of the displacement vector', argstr='-magnitude')
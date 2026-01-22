import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ConvertDsetInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to ConvertDset', argstr='-input %s', position=-2, mandatory=True, exists=True)
    out_file = File(desc='output file for ConvertDset', argstr='-prefix %s', position=-1, mandatory=True)
    out_type = traits.Enum(('niml', 'niml_asc', 'niml_bi', '1D', '1Dp', '1Dpt', 'gii', 'gii_asc', 'gii_b64', 'gii_b64gz'), desc='output type', argstr='-o_%s', mandatory=True, position=0)
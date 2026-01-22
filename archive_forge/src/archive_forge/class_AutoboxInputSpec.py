import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class AutoboxInputSpec(AFNICommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='-input %s', desc='input file', copyfile=False)
    padding = traits.Int(argstr='-npad %d', desc='Number of extra voxels to pad on each side of box')
    out_file = File(argstr='-prefix %s', name_source='in_file', name_template='%s_autobox')
    no_clustering = traits.Bool(argstr='-noclust', desc="Don't do any clustering to find box. Any non-zero voxel will be preserved in the cropped volume. The default method uses some clustering to find the cropping box, and will clip off small isolated blobs.")
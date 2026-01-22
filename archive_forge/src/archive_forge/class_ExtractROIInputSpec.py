import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ExtractROIInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='%s', position=0, desc='input file', mandatory=True)
    roi_file = File(argstr='%s', position=1, desc='output file', genfile=True, hash_files=False)
    x_min = traits.Int(argstr='%d', position=2)
    x_size = traits.Int(argstr='%d', position=3)
    y_min = traits.Int(argstr='%d', position=4)
    y_size = traits.Int(argstr='%d', position=5)
    z_min = traits.Int(argstr='%d', position=6)
    z_size = traits.Int(argstr='%d', position=7)
    t_min = traits.Int(argstr='%d', position=8)
    t_size = traits.Int(argstr='%d', position=9)
    _crop_xor = ['x_min', 'x_size', 'y_min', 'y_size', 'z_min', 'z_size', 't_min', 't_size']
    crop_list = traits.List(traits.Tuple(traits.Int, traits.Int), argstr='%s', position=2, xor=_crop_xor, desc='list of two tuples specifying crop options')
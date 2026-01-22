import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class RobustFOVInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, desc='input filename', argstr='-i %s', position=0, mandatory=True)
    out_roi = File(desc='ROI volume output name', argstr='-r %s', name_source=['in_file'], hash_files=False, name_template='%s_ROI')
    brainsize = traits.Int(desc='size of brain in z-dimension (default 170mm/150mm)', argstr='-b %d')
    out_transform = File(desc='Transformation matrix in_file to out_roi output name', argstr='-m %s', name_source=['in_file'], hash_files=False, name_template='%s_to_ROI')
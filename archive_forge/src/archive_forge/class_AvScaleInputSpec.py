import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class AvScaleInputSpec(CommandLineInputSpec):
    all_param = traits.Bool(False, argstr='--allparams')
    mat_file = File(exists=True, argstr='%s', desc='mat file to read', position=-2)
    ref_file = File(exists=True, argstr='%s', position=-1, desc='reference file to get center of rotation')
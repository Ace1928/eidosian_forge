import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PowerSpectrumInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, desc='input 4D file to estimate the power spectrum', argstr='%s', position=0, mandatory=True)
    out_file = File(desc='name of output 4D file for power spectrum', argstr='%s', position=1, genfile=True, hash_files=False)
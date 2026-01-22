import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class Vest2TextInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, desc='matrix data stored in the format used by FSL tools', argstr='%s', position=0)
    out_file = File('design.txt', usedefault=True, desc='file name to store text output from matrix', argstr='%s', position=1)
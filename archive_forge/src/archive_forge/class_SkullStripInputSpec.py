import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class SkullStripInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dSkullStrip', argstr='-input %s', position=1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_skullstrip', desc='output image file name', argstr='-prefix %s', name_source='in_file')
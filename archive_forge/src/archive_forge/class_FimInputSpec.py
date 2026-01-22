import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class FimInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dfim+', argstr='-input %s', position=1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_fim', desc='output image file name', argstr='-bucket %s', name_source='in_file')
    ideal_file = File(desc='ideal time series file name', argstr='-ideal_file %s', position=2, mandatory=True, exists=True)
    fim_thr = traits.Float(desc='fim internal mask threshold value', argstr='-fim_thr %f', position=3)
    out = Str(desc='Flag to output the specified parameter', argstr='-out %s', position=4)
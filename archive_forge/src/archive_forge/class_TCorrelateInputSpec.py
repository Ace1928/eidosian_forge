import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TCorrelateInputSpec(AFNICommandInputSpec):
    xset = File(desc='input xset', argstr='%s', position=-2, mandatory=True, exists=True, copyfile=False)
    yset = File(desc='input yset', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_tcorr', desc='output image file name', argstr='-prefix %s', name_source='xset')
    pearson = traits.Bool(desc='Correlation is the normal Pearson correlation coefficient', argstr='-pearson')
    polort = traits.Int(desc='Remove polynomical trend of order m', argstr='-polort %d')
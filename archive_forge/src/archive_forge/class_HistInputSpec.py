import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class HistInputSpec(CommandLineInputSpec):
    in_file = File(desc='input file to 3dHist', argstr='-input %s', position=1, mandatory=True, exists=True, copyfile=False)
    out_file = File(desc='Write histogram to niml file with this prefix', name_template='%s_hist', keep_extension=False, argstr='-prefix %s', name_source=['in_file'])
    showhist = traits.Bool(False, usedefault=True, desc='write a text visual histogram', argstr='-showhist')
    out_show = File(name_template='%s_hist.out', desc='output image file name', keep_extension=False, argstr='> %s', name_source='in_file', position=-1)
    mask = File(desc='matrix to align input file', argstr='-mask %s', exists=True)
    nbin = traits.Int(desc='number of bins', argstr='-nbin %d')
    max_value = traits.Float(argstr='-max %f', desc='maximum intensity value')
    min_value = traits.Float(argstr='-min %f', desc='minimum intensity value')
    bin_width = traits.Float(argstr='-binwidth %f', desc='bin width')
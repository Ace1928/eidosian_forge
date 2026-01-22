import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TSmoothInputSpec(AFNICommandInputSpec):
    in_file = File(desc='input file to 3dTSmooth', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_smooth', desc='output file from 3dTSmooth', argstr='-prefix %s', name_source='in_file')
    datum = traits.Str(desc='Sets the data type of the output dataset', argstr='-datum %s')
    lin = traits.Bool(desc='3 point linear filter: :math:`0.15\\,a + 0.70\\,b + 0.15\\,c` [This is the default smoother]', argstr='-lin')
    med = traits.Bool(desc='3 point median filter: median(a,b,c)', argstr='-med')
    osf = traits.Bool(desc='3 point order statistics filter::math:`0.15\\,min(a,b,c) + 0.70\\,median(a,b,c) + 0.15\\,max(a,b,c)`', argstr='-osf')
    lin3 = traits.Int(desc="3 point linear filter: :math:`0.5\\,(1-m)\\,a + m\\,b + 0.5\\,(1-m)\\,c`. Here, 'm' is a number strictly between 0 and 1.", argstr='-3lin %d')
    hamming = traits.Int(argstr='-hamming %d', desc='Use N point Hamming windows. (N must be odd and bigger than 1.)')
    blackman = traits.Int(argstr='-blackman %d', desc='Use N point Blackman windows. (N must be odd and bigger than 1.)')
    custom = File(argstr='-custom %s', desc='odd # of coefficients must be in a single column in ASCII file')
    adaptive = traits.Int(argstr='-adaptive %d', desc='use adaptive mean filtering of width N (where N must be odd and bigger than 3).')
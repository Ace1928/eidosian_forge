import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class MeansInputSpec(AFNICommandInputSpec):
    in_file_a = File(desc='input file to 3dMean', argstr='%s', position=-2, mandatory=True, exists=True)
    in_file_b = File(desc='another input file to 3dMean', argstr='%s', position=-1, exists=True)
    datum = traits.Str(desc='Sets the data type of the output dataset', argstr='-datum %s')
    out_file = File(name_template='%s_mean', desc='output image file name', argstr='-prefix %s', name_source='in_file_a')
    scale = Str(desc='scaling of output', argstr='-%sscale')
    non_zero = traits.Bool(desc='use only non-zero values', argstr='-non_zero')
    std_dev = traits.Bool(desc='calculate std dev', argstr='-stdev')
    sqr = traits.Bool(desc='mean square instead of value', argstr='-sqr')
    summ = traits.Bool(desc='take sum, (not average)', argstr='-sum')
    count = traits.Bool(desc='compute count of non-zero voxels', argstr='-count')
    mask_inter = traits.Bool(desc='create intersection mask', argstr='-mask_inter')
    mask_union = traits.Bool(desc='create union mask', argstr='-mask_union')
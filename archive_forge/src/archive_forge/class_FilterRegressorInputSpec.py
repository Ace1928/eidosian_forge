import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FilterRegressorInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, argstr='-i %s', desc='input file name (4D image)', mandatory=True, position=1)
    out_file = File(argstr='-o %s', desc='output file name for the filtered data', genfile=True, position=2, hash_files=False)
    design_file = File(exists=True, argstr='-d %s', position=3, mandatory=True, desc='name of the matrix with time courses (e.g. GLM design or MELODIC mixing matrix)')
    filter_columns = traits.List(traits.Int, argstr="-f '%s'", xor=['filter_all'], mandatory=True, position=4, desc='(1-based) column indices to filter out of the data')
    filter_all = traits.Bool(mandatory=True, argstr="-f '%s'", xor=['filter_columns'], position=4, desc='use all columns in the design file in denoising')
    mask = File(exists=True, argstr='-m %s', desc='mask image file name')
    var_norm = traits.Bool(argstr='--vn', desc='perform variance-normalization on data')
    out_vnscales = traits.Bool(argstr='--out_vnscales', desc='output scaling factors for variance normalization')
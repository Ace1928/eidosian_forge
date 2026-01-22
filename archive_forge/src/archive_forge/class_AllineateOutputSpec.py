import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class AllineateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image file name')
    out_matrix = File(exists=True, desc='matrix to align input file')
    out_param_file = File(exists=True, desc='warp parameters')
    out_weight_file = File(exists=True, desc='weight volume')
    allcostx = File(desc='Compute and print ALL available cost functionals for the un-warped inputs')
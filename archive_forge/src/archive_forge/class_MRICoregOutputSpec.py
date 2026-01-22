import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class MRICoregOutputSpec(TraitedSpec):
    out_reg_file = File(exists=True, desc='output registration file')
    out_lta_file = File(exists=True, desc='output LTA-style registration file')
    out_params_file = File(exists=True, desc='output parameters file')
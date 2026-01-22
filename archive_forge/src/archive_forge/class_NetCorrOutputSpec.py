import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class NetCorrOutputSpec(TraitedSpec):
    out_corr_matrix = File(desc='output correlation matrix between ROIs written to a text file with .netcc suffix')
    out_corr_maps = traits.List(File(), desc='output correlation maps in Pearson and/or Z-scores')
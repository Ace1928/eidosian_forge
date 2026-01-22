import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
def _make_inv_file(self):
    """makes filename to hold inverse transform if not specified"""
    invmat = fname_presuffix(self.inputs.mat, prefix='inverse_')
    return invmat
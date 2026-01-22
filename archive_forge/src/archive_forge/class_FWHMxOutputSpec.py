import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class FWHMxOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output file')
    out_subbricks = File(exists=True, desc='output file (subbricks)')
    out_detrend = File(desc='output file, detrended')
    fwhm = traits.Either(traits.Tuple(traits.Float(), traits.Float(), traits.Float()), traits.Tuple(traits.Float(), traits.Float(), traits.Float(), traits.Float()), desc='FWHM along each axis')
    acf_param = traits.Either(traits.Tuple(traits.Float(), traits.Float(), traits.Float()), traits.Tuple(traits.Float(), traits.Float(), traits.Float(), traits.Float()), desc='fitted ACF model parameters')
    out_acf = File(exists=True, desc='output acf file')
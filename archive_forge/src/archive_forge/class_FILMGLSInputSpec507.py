import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FILMGLSInputSpec507(FILMGLSInputSpec505):
    threshold = traits.Float(default_value=-1000.0, argstr='--thr=%f', position=-1, usedefault=True, desc='threshold')
    tcon_file = File(exists=True, argstr='--con=%s', desc='contrast file containing T-contrasts')
    fcon_file = File(exists=True, argstr='--fcon=%s', desc='contrast file containing F-contrasts')
    mode = traits.Enum('volumetric', 'surface', argstr='--mode=%s', desc='Type of analysis to be done')
    surface = File(exists=True, argstr='--in2=%s', desc='input surface for autocorr smoothing in surface-based analyses')
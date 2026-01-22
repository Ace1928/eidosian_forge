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
class SmoothEstimateInputSpec(FSLCommandInputSpec):
    dof = traits.Int(argstr='--dof=%d', mandatory=True, xor=['zstat_file'], desc='number of degrees of freedom')
    mask_file = File(argstr='--mask=%s', exists=True, mandatory=True, desc='brain mask volume')
    residual_fit_file = File(argstr='--res=%s', exists=True, requires=['dof'], desc='residual-fit image file')
    zstat_file = File(argstr='--zstat=%s', exists=True, xor=['dof'], desc='zstat image file')
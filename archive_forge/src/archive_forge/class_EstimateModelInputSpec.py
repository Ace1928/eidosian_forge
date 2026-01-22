import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class EstimateModelInputSpec(SPMCommandInputSpec):
    spm_mat_file = File(exists=True, field='spmmat', copyfile=True, mandatory=True, desc='Absolute path to SPM.mat')
    estimation_method = traits.Dict(traits.Enum('Classical', 'Bayesian2', 'Bayesian'), field='method', mandatory=True, desc='Dictionary of either Classical: 1, Bayesian: 1, or Bayesian2: 1 (dict)')
    write_residuals = traits.Bool(field='write_residuals', desc='Write individual residual images')
    flags = traits.Dict(desc='Additional arguments')
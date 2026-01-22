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
class FILMGLSOutputSpec(TraitedSpec):
    param_estimates = OutputMultiPath(File(exists=True), desc='Parameter estimates for each column of the design matrix')
    residual4d = File(exists=True, desc='Model fit residual mean-squared error for each time point')
    dof_file = File(exists=True, desc='degrees of freedom')
    sigmasquareds = File(exists=True, desc='summary of residuals, See Woolrich, et. al., 2001')
    results_dir = Directory(exists=True, desc='directory storing model estimation output')
    corrections = File(exists=True, desc='statistical corrections used within FILM modeling')
    thresholdac = File(exists=True, desc='The FILM autocorrelation parameters')
    logfile = File(exists=True, desc='FILM run logfile')
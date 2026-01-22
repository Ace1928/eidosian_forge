import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class FactorialDesignOutputSpec(TraitedSpec):
    spm_mat_file = File(exists=True, desc='SPM mat file')
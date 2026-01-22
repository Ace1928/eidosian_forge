import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class BBRegisterInputSpec6(BBRegisterInputSpec):
    init = traits.Enum('coreg', 'rr', 'spm', 'fsl', 'header', 'best', argstr='--init-%s', xor=['init_reg_file'], desc='initialize registration with mri_coreg, spm, fsl, or header')
    init_reg_file = File(exists=True, argstr='--init-reg %s', desc='existing registration file', xor=['init'])
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
class BBRegisterOutputSpec(TraitedSpec):
    out_reg_file = File(exists=True, desc='Output registration file')
    out_fsl_file = File(exists=True, desc='Output FLIRT-style registration file')
    out_lta_file = File(exists=True, desc='Output LTA-style registration file')
    min_cost_file = File(exists=True, desc='Output registration minimum cost file')
    init_cost_file = File(exists=True, desc='Output initial registration cost file')
    registered_file = File(exists=True, desc='Registered and resampled source file')
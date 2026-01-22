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
def _get_outfilename(self):
    if isdefined(self.inputs.resampled_file):
        outfile = self.inputs.resampled_file
    else:
        outfile = fname_presuffix(self.inputs.in_file, newpath=os.getcwd(), suffix='_resample')
    return outfile
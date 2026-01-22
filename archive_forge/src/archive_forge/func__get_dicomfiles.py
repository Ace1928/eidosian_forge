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
def _get_dicomfiles(self):
    """validate fsl bet options
        if set to None ignore
        """
    return glob(os.path.abspath(os.path.join(self.inputs.dicom_dir, '*-1.dcm')))
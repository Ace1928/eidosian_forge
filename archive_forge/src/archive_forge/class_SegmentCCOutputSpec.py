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
class SegmentCCOutputSpec(TraitedSpec):
    out_file = File(exists=False, desc='Output segmentation uncluding corpus collosum')
    out_rotation = File(exists=False, desc='Output lta rotation file')
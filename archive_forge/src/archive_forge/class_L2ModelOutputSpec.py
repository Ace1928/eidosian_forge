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
class L2ModelOutputSpec(TraitedSpec):
    design_mat = File(exists=True, desc='design matrix file')
    design_con = File(exists=True, desc='design contrast file')
    design_grp = File(exists=True, desc='design group file')
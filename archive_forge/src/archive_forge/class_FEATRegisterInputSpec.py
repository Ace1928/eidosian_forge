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
class FEATRegisterInputSpec(BaseInterfaceInputSpec):
    feat_dirs = InputMultiPath(Directory(exists=True), desc='Lower level feat dirs', mandatory=True)
    reg_image = File(exists=True, desc='image to register to (will be treated as standard)', mandatory=True)
    reg_dof = traits.Int(12, desc='registration degrees of freedom', usedefault=True)
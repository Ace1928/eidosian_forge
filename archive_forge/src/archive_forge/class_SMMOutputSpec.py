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
class SMMOutputSpec(TraitedSpec):
    null_p_map = File(exists=True)
    activation_p_map = File(exists=True)
    deactivation_p_map = File(exists=True)
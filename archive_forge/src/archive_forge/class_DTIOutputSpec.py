import numpy as np
import nibabel as nb
from ... import logging
from ..base import TraitedSpec, File, isdefined
from .base import DipyDiffusionInterface, DipyBaseInterfaceInputSpec
class DTIOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    fa_file = File(exists=True)
    md_file = File(exists=True)
    rd_file = File(exists=True)
    ad_file = File(exists=True)
    color_fa_file = File(exists=True)
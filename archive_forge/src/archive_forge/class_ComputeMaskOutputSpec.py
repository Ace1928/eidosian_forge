import os
import nibabel as nb
import numpy as np
from ...utils.filemanip import split_filename, fname_presuffix
from .base import NipyBaseInterface, have_nipy
from ..base import (
class ComputeMaskOutputSpec(TraitedSpec):
    brain_mask = File(exists=True)
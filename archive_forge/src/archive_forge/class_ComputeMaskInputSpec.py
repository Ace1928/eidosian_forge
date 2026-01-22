import os
import nibabel as nb
import numpy as np
from ...utils.filemanip import split_filename, fname_presuffix
from .base import NipyBaseInterface, have_nipy
from ..base import (
class ComputeMaskInputSpec(BaseInterfaceInputSpec):
    mean_volume = File(exists=True, mandatory=True, desc='mean EPI image, used to compute the threshold for the mask')
    reference_volume = File(exists=True, desc='reference volume used to compute the mask. If none is give, the mean volume is used.')
    m = traits.Float(desc='lower fraction of the histogram to be discarded')
    M = traits.Float(desc='upper fraction of the histogram to be discarded')
    cc = traits.Bool(desc='Keep only the largest connected component')
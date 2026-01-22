import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class OverlapInputSpec(BaseInterfaceInputSpec):
    volume1 = File(exists=True, mandatory=True, desc='Has to have the same dimensions as volume2.')
    volume2 = File(exists=True, mandatory=True, desc='Has to have the same dimensions as volume1.')
    mask_volume = File(exists=True, desc='calculate overlap only within this mask.')
    bg_overlap = traits.Bool(False, usedefault=True, mandatory=True, desc='consider zeros as a label')
    out_file = File('diff.nii', usedefault=True)
    weighting = traits.Enum('none', 'volume', 'squared_vol', usedefault=True, desc="'none': no class-overlap weighting is performed. 'volume': computed class-overlaps are weighted by class volume 'squared_vol': computed class-overlaps are weighted by the squared volume of the class")
    vol_units = traits.Enum('voxel', 'mm', mandatory=True, usedefault=True, desc='units for volumes')
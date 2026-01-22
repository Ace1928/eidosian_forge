import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class ThresholdStatisticsOutputSpec(TraitedSpec):
    voxelwise_P_Bonf = traits.Float()
    voxelwise_P_RF = traits.Float()
    voxelwise_P_uncor = traits.Float()
    voxelwise_P_FDR = traits.Float()
    clusterwise_P_RF = traits.Float()
    clusterwise_P_FDR = traits.Float()
import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EpiRegOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='unwarped and coregistered epi input')
    out_1vol = File(exists=True, desc='unwarped and coregistered single volume')
    fmap2str_mat = File(exists=True, desc='rigid fieldmap-to-structural transform')
    fmap2epi_mat = File(exists=True, desc='rigid fieldmap-to-epi transform')
    fmap_epi = File(exists=True, desc='fieldmap in epi space')
    fmap_str = File(exists=True, desc='fieldmap in structural space')
    fmapmag_str = File(exists=True, desc='fieldmap magnitude image in structural space')
    epi2str_inv = File(exists=True, desc='rigid structural-to-epi transform')
    epi2str_mat = File(exists=True, desc='rigid epi-to-structural transform')
    shiftmap = File(exists=True, desc='shiftmap in epi space')
    fullwarp = File(exists=True, desc='warpfield to unwarp epi and transform into                     structural space')
    wmseg = File(exists=True, desc='white matter segmentation used in flirt bbr')
    seg = File(exists=True, desc='white matter, gray matter, csf segmentation')
    wmedge = File(exists=True, desc='white matter edges for visualization')
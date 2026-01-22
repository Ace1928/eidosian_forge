import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class ANTSOutputSpec(TraitedSpec):
    affine_transform = File(exists=True, desc='Affine transform file')
    warp_transform = File(exists=True, desc='Warping deformation field')
    inverse_warp_transform = File(exists=True, desc='Inverse warping deformation field')
    metaheader = File(exists=True, desc='VTK metaheader .mhd file')
    metaheader_raw = File(exists=True, desc='VTK metaheader .raw file')
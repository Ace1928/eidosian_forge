import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class NIfTIDT2CaminoInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='-inputfile %s', mandatory=True, position=1, desc='A NIFTI-1 dataset containing diffusion tensors. The tensors are assumed to be in lower-triangular order as specified by the NIFTI standard for the storage of symmetric matrices. This file should be either a .nii or a .hdr file.')
    s0_file = File(argstr='-s0 %s', exists=True, desc='File containing the unweighted signal for each voxel, may be a raw binary file (specify type with -inputdatatype) or a supported image file.')
    lns0_file = File(argstr='-lns0 %s', exists=True, desc='File containing the log of the unweighted signal for each voxel, may be a raw binary file (specify type with -inputdatatype) or a supported image file.')
    bgmask = File(argstr='-bgmask %s', exists=True, desc='Binary valued brain / background segmentation, may be a raw binary file (specify type with -maskdatatype) or a supported image file.')
    scaleslope = traits.Float(argstr='-scaleslope %s', desc='A value v in the diffusion tensor is scaled to v * s + i. This is applied after any scaling specified by the input image. Default is 1.0.')
    scaleinter = traits.Float(argstr='-scaleinter %s', desc='A value v in the diffusion tensor is scaled to v * s + i. This is applied after any scaling specified by the input image. Default is 0.0.')
    uppertriangular = traits.Bool(argstr='-uppertriangular %s', desc='Specifies input in upper-triangular (VTK style) order.')
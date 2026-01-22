import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProbTrackXInputSpec(ProbTrackXBaseInputSpec):
    mode = traits.Enum('simple', 'two_mask_symm', 'seedmask', desc='options: simple (single seed voxel), seedmask (mask of seed voxels), twomask_symm (two bet binary masks)', argstr='--mode=%s', genfile=True)
    mask2 = File(exists=True, desc='second bet binary mask (in diffusion space) in twomask_symm mode', argstr='--mask2=%s')
    mesh = File(exists=True, desc='Freesurfer-type surface descriptor (in ascii format)', argstr='--mesh=%s')
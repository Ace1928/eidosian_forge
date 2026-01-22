import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class DistanceMapInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='--in=%s', desc='image to calculate distance values for')
    mask_file = File(exists=True, argstr='--mask=%s', desc='binary mask to constrain calculations')
    invert_input = traits.Bool(argstr='--invert', desc='invert input image')
    local_max_file = traits.Either(traits.Bool, File, argstr='--localmax=%s', desc='write an image of the local maxima', hash_files=False)
    distance_map = File(genfile=True, argstr='--out=%s', desc='distance map to write', hash_files=False)
import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ApplyMaskInputSpec(FSTraitedSpec):
    in_file = File(exists=True, mandatory=True, position=-3, argstr='%s', desc='input image (will be masked)')
    mask_file = File(exists=True, mandatory=True, position=-2, argstr='%s', desc='image defining mask space')
    out_file = File(name_source=['in_file'], name_template='%s_masked', hash_files=True, keep_extension=True, position=-1, argstr='%s', desc='final image to write')
    xfm_file = File(exists=True, argstr='-xform %s', desc='LTA-format transformation matrix to align mask with input')
    invert_xfm = traits.Bool(argstr='-invert', desc='invert transformation')
    xfm_source = File(exists=True, argstr='-lta_src %s', desc='image defining transform source space')
    xfm_target = File(exists=True, argstr='-lta_dst %s', desc='image defining transform target space')
    use_abs = traits.Bool(argstr='-abs', desc='take absolute value of mask before applying')
    mask_thresh = traits.Float(argstr='-T %.4f', desc='threshold mask before applying')
    keep_mask_deletion_edits = traits.Bool(argstr='-keep_mask_deletion_edits', desc='transfer voxel-deletion edits (voxels=1) from mask to out vol')
    transfer = traits.Int(argstr='-transfer %d', desc='transfer only voxel value # from mask to out')
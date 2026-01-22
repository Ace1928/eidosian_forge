import os
from ...utils.filemanip import split_filename
from ..base import (
class LinReconInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=1, desc='voxel-order data filename')
    scheme_file = File(exists=True, argstr='%s', mandatory=True, position=2, desc='Specifies the scheme file for the diffusion MRI data')
    qball_mat = File(exists=True, argstr='%s', mandatory=True, position=3, desc='Linear transformation matrix.')
    normalize = traits.Bool(argstr='-normalize', desc='Normalize the measurements and discard the zero measurements before the linear transform.')
    log = traits.Bool(argstr='-log', desc='Transform the log measurements rather than the measurements themselves')
    bgmask = File(exists=True, argstr='-bgmask %s', desc='background mask')
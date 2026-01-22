import os
from ...utils.filemanip import split_filename
from ..base import (
class DTIFitInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=1, desc='voxel-order data filename')
    bgmask = File(argstr='-bgmask %s', exists=True, desc='Provides the name of a file containing a background mask computed using, for example, FSL bet2 program. The mask file contains zero in background voxels and non-zero in foreground.')
    scheme_file = File(exists=True, argstr='%s', mandatory=True, position=2, desc='Camino scheme file (b values / vectors, see camino.fsl2scheme)')
    non_linear = traits.Bool(argstr='-nonlinear', position=3, desc='Use non-linear fitting instead of the default linear regression to the log measurements. ')
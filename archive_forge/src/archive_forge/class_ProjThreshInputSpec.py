import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProjThreshInputSpec(FSLCommandInputSpec):
    in_files = traits.List(File(exists=True), argstr='%s', desc='a list of input volumes', mandatory=True, position=0)
    threshold = traits.Int(argstr='%d', desc='threshold indicating minimum number of seed voxels entering this mask region', mandatory=True, position=1)
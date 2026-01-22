import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class XFibres5(FSLXCommand):
    """
    Perform model parameters estimation for local (voxelwise) diffusion
    parameters
    """
    _cmd = 'xfibres'
    input_spec = XFibres5InputSpec
    output_spec = FSLXCommandOutputSpec
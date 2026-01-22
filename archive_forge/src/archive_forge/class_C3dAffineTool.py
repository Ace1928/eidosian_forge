import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging
class C3dAffineTool(SEMLikeCommandLine):
    """Converts fsl-style Affine registration into ANTS compatible itk format

    Example
    =======

    >>> from nipype.interfaces.c3 import C3dAffineTool
    >>> c3 = C3dAffineTool()
    >>> c3.inputs.source_file = 'cmatrix.mat'
    >>> c3.inputs.itk_transform = 'affine.txt'
    >>> c3.inputs.fsl2ras = True
    >>> c3.cmdline
    'c3d_affine_tool -src cmatrix.mat -fsl2ras -oitk affine.txt'
    """
    input_spec = C3dAffineToolInputSpec
    output_spec = C3dAffineToolOutputSpec
    _cmd = 'c3d_affine_tool'
    _outputs_filenames = {'itk_transform': 'affine.txt'}
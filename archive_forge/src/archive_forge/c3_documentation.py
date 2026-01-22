import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging

    Convert3d is a command-line tool for converting 3D (or 4D) images between
    common file formats. The tool also includes a growing list of commands for
    image manipulation, such as thresholding and resampling. The tool can also
    be used to obtain information about image files. More information on
    Convert3d can be found at:
    https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md


    Example
    =======

    >>> from nipype.interfaces.c3 import C3d
    >>> c3 = C3d()
    >>> c3.inputs.in_file = "T1.nii"
    >>> c3.inputs.pix_type = "short"
    >>> c3.inputs.out_file = "T1.img"
    >>> c3.cmdline
    'c3d T1.nii -type short -o T1.img'
    >>> c3.inputs.is_4d = True
    >>> c3.inputs.in_file = "epi.nii"
    >>> c3.inputs.out_file = "epi.img"
    >>> c3.cmdline
    'c4d epi.nii -type short -o epi.img'
    
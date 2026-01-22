import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class BinaryMathsInputInteger(MathsInput):
    """Input Spec for seg_maths Binary operations that require integer."""
    operation = traits.Enum('dil', 'ero', 'tp', 'equal', 'pad', 'crop', mandatory=True, argstr='-%s', position=4, desc='Operation to perform:\n\n    * equal - <int> - Get voxels equal to <int>\n    * dil - <int>  - Dilate the image <int> times (in voxels).\n    * ero - <int> - Erode the image <int> times (in voxels).\n    * tp - <int> - Extract time point <int>\n    * crop - <int> - Crop <int> voxels around each 3D volume.\n    * pad - <int> -  Pad <int> voxels with NaN value around each 3D volume.\n\n')
    operand_value = traits.Int(argstr='%d', mandatory=True, position=5, desc='int value to perform operation with')
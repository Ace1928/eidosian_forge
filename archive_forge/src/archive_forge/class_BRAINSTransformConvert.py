import os
from ...base import (
class BRAINSTransformConvert(SEMLikeCommandLine):
    """title: BRAINS Transform Convert

    category: Utilities.BRAINS

    description: Convert ITK transforms to higher order transforms

    version: 1.0

    documentation-url: A utility to convert between transform file formats.

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Hans J. Johnson,Kent Williams, Ali Ghayoor
    """
    input_spec = BRAINSTransformConvertInputSpec
    output_spec = BRAINSTransformConvertOutputSpec
    _cmd = ' BRAINSTransformConvert '
    _outputs_filenames = {'displacementVolume': 'displacementVolume.nii', 'outputTransform': 'outputTransform.mat'}
    _redirect_x = False
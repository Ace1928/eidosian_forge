import os
from ...base import (
class BRAINSResize(SEMLikeCommandLine):
    """title: Resize Image (BRAINS)

    category: Registration

    description: This program is useful for downsampling an image by a constant scale factor.

    version: 3.0.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Hans Johnson.

    acknowledgements: The development of this tool was supported by funding from grants NS050568 and NS40068 from the National Institute of Neurological Disorders and Stroke and grants MH31593, MH40856, from the National Institute of Mental Health.
    """
    input_spec = BRAINSResizeInputSpec
    output_spec = BRAINSResizeOutputSpec
    _cmd = ' BRAINSResize '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False
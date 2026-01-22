import os
from ...base import (
class DilateImage(SEMLikeCommandLine):
    """title: Dilate Image

    category: Filtering.FeatureDetection

    description: Uses mathematical morphology to dilate the input images.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Mark Scully and Jeremy Bockholt.
    """
    input_spec = DilateImageInputSpec
    output_spec = DilateImageOutputSpec
    _cmd = ' DilateImage '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False
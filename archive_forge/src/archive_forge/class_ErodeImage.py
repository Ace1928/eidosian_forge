import os
from ...base import (
class ErodeImage(SEMLikeCommandLine):
    """title: Erode Image

    category: Filtering.FeatureDetection

    description: Uses mathematical morphology to erode the input images.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Mark Scully and Jeremy Bockholt.
    """
    input_spec = ErodeImageInputSpec
    output_spec = ErodeImageOutputSpec
    _cmd = ' ErodeImage '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False
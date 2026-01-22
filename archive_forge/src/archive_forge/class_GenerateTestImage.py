import os
from ...base import (
class GenerateTestImage(SEMLikeCommandLine):
    """title: DownSampleImage

    category: Filtering.FeatureDetection

    description: Down sample image for testing

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Eun Young Kim
    """
    input_spec = GenerateTestImageInputSpec
    output_spec = GenerateTestImageOutputSpec
    _cmd = ' GenerateTestImage '
    _outputs_filenames = {'outputVolume': 'outputVolume'}
    _redirect_x = False
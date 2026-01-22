import os
from ...base import (
class GenerateSummedGradientImage(SEMLikeCommandLine):
    """title: GenerateSummedGradient

    category: Filtering.FeatureDetection

    description: Automatic FeatureImages using neural networks

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Greg Harris, Eun Young Kim
    """
    input_spec = GenerateSummedGradientImageInputSpec
    output_spec = GenerateSummedGradientImageOutputSpec
    _cmd = ' GenerateSummedGradientImage '
    _outputs_filenames = {'outputFileName': 'outputFileName'}
    _redirect_x = False
import os
from ...base import (
class ShuffleVectorsModule(SEMLikeCommandLine):
    """title: ShuffleVectors

    category: Utilities.BRAINS

    description: Automatic Segmentation using neural networks

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Hans Johnson
    """
    input_spec = ShuffleVectorsModuleInputSpec
    output_spec = ShuffleVectorsModuleOutputSpec
    _cmd = ' ShuffleVectorsModule '
    _outputs_filenames = {'outputVectorFileBaseName': 'outputVectorFileBaseName'}
    _redirect_x = False
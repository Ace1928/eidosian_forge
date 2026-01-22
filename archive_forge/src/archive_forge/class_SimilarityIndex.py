import os
from ...base import (
class SimilarityIndex(SEMLikeCommandLine):
    """title: BRAINSCut:SimilarityIndexComputation

    category: BRAINS.Segmentation

    description: Automatic analysis of BRAINSCut Output

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Eunyoung Regin Kim
    """
    input_spec = SimilarityIndexInputSpec
    output_spec = SimilarityIndexOutputSpec
    _cmd = ' SimilarityIndex '
    _outputs_filenames = {}
    _redirect_x = False
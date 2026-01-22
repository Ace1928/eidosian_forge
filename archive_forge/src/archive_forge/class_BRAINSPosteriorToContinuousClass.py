import os
from ...base import (
class BRAINSPosteriorToContinuousClass(SEMLikeCommandLine):
    """title: Tissue Classification

    category: BRAINS.Classify

    description: This program will generate an 8-bit continuous tissue classified image based on BRAINSABC posterior images.

    version: 3.0

    documentation-url: http://www.nitrc.org/plugins/mwiki/index.php/brains:BRAINSClassify

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Vincent A. Magnotta

    acknowledgements: Funding for this work was provided by NIH/NINDS award NS050568
    """
    input_spec = BRAINSPosteriorToContinuousClassInputSpec
    output_spec = BRAINSPosteriorToContinuousClassOutputSpec
    _cmd = ' BRAINSPosteriorToContinuousClass '
    _outputs_filenames = {'outputVolume': 'outputVolume'}
    _redirect_x = False
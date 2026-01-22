import os
from ...base import (
class BRAINSLinearModelerEPCA(SEMLikeCommandLine):
    """title: Landmark Linear Modeler (BRAINS)

    category: Utilities.BRAINS

    description: Training linear model using EPCA. Implementation based on my MS thesis, "A METHOD FOR AUTOMATED LANDMARK CONSTELLATION DETECTION USING EVOLUTIONARY PRINCIPAL COMPONENTS AND STATISTICAL SHAPE MODELS"

    version: 1.0

    documentation-url: http://www.nitrc.org/projects/brainscdetector/
    """
    input_spec = BRAINSLinearModelerEPCAInputSpec
    output_spec = BRAINSLinearModelerEPCAOutputSpec
    _cmd = ' BRAINSLinearModelerEPCA '
    _outputs_filenames = {}
    _redirect_x = False
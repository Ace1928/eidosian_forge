import os
from ...base import (
class BRAINSConstellationModeler(SEMLikeCommandLine):
    """title: Generate Landmarks Model (BRAINS)

    category: Utilities.BRAINS

    description: Train up a model for BRAINSConstellationDetector
    """
    input_spec = BRAINSConstellationModelerInputSpec
    output_spec = BRAINSConstellationModelerOutputSpec
    _cmd = ' BRAINSConstellationModeler '
    _outputs_filenames = {'outputModel': 'outputModel.mdl', 'resultsDir': 'resultsDir'}
    _redirect_x = False
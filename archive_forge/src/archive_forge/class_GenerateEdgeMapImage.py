import os
from ...base import (
class GenerateEdgeMapImage(SEMLikeCommandLine):
    """title: GenerateEdgeMapImage

    category: BRAINS.Utilities

    description: Automatic edgemap generation for edge-guided super-resolution reconstruction

    version: 1.0

    contributor: Ali Ghayoor
    """
    input_spec = GenerateEdgeMapImageInputSpec
    output_spec = GenerateEdgeMapImageOutputSpec
    _cmd = ' GenerateEdgeMapImage '
    _outputs_filenames = {'outputEdgeMap': 'outputEdgeMap', 'outputMaximumGradientImage': 'outputMaximumGradientImage'}
    _redirect_x = False
import os
from ...base import (
class GradientAnisotropicDiffusionImageFilter(SEMLikeCommandLine):
    """title: GradientAnisopropicDiffusionFilter

    category: Filtering.FeatureDetection

    description: Image Smoothing using Gradient Anisotropic Diffuesion Filer

    contributor: This tool was developed by Eun Young Kim by modifying ITK Example
    """
    input_spec = GradientAnisotropicDiffusionImageFilterInputSpec
    output_spec = GradientAnisotropicDiffusionImageFilterOutputSpec
    _cmd = ' GradientAnisotropicDiffusionImageFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False
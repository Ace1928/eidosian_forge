from nipype.interfaces.base import (
import os
class DiffusionWeightedVolumeMasking(SEMLikeCommandLine):
    """title: Diffusion Weighted Volume Masking

    category: Diffusion.Diffusion Weighted Images

    description: <p>Performs a mask calculation from a diffusion weighted (DW) image.</p><p>Starting from a dw image, this module computes the baseline image averaging all the images without diffusion weighting and then applies the otsu segmentation algorithm in order to produce a mask. this mask can then be used when estimating the diffusion tensor (dt) image, not to estimate tensors all over the volume.</p>

    version: 0.1.0.$Revision: 1892 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/DiffusionWeightedMasking

    license: slicer3

    contributor: Demian Wassermann (SPL, BWH)
    """
    input_spec = DiffusionWeightedVolumeMaskingInputSpec
    output_spec = DiffusionWeightedVolumeMaskingOutputSpec
    _cmd = 'DiffusionWeightedVolumeMasking '
    _outputs_filenames = {'outputBaseline': 'outputBaseline.nii', 'thresholdMask': 'thresholdMask.nii'}
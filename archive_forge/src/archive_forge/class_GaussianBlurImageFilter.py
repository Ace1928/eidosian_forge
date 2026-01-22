from nipype.interfaces.base import (
import os
class GaussianBlurImageFilter(SEMLikeCommandLine):
    """title: Gaussian Blur Image Filter

    category: Filtering.Denoising

    description: Apply a gaussian blur to an image

    version: 0.1.0.$Revision: 1.1 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/GaussianBlurImageFilter

    contributor: Julien Jomier (Kitware), Stephen Aylward (Kitware)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = GaussianBlurImageFilterInputSpec
    output_spec = GaussianBlurImageFilterOutputSpec
    _cmd = 'GaussianBlurImageFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
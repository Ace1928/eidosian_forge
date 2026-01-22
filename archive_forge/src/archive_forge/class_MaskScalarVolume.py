from nipype.interfaces.base import (
import os
class MaskScalarVolume(SEMLikeCommandLine):
    """title: Mask Scalar Volume

    category: Filtering.Arithmetic

    description: Masks two images. The output image is set to 0 everywhere except where the chosen label from the mask volume is present, at which point it will retain it's original values. Although all image types are supported on input, only signed types are produced. The two images do not have to have the same dimensions.

    version: 0.1.0.$Revision: 8595 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/Mask

    contributor: Nicole Aucoin (SPL, BWH), Ron Kikinis (SPL, BWH)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = MaskScalarVolumeInputSpec
    output_spec = MaskScalarVolumeOutputSpec
    _cmd = 'MaskScalarVolume '
    _outputs_filenames = {'OutputVolume': 'OutputVolume.nii'}
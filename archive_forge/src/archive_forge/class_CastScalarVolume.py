from nipype.interfaces.base import (
import os
class CastScalarVolume(SEMLikeCommandLine):
    """title: Cast Scalar Volume

    category: Filtering.Arithmetic

    description: Cast a volume to a given data type.
    Use at your own risk when casting an input volume into a lower precision type!
    Allows casting to the same type as the input volume.

    version: 0.1.0.$Revision: 2104 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/Cast

    contributor: Nicole Aucoin (SPL, BWH), Ron Kikinis (SPL, BWH)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = CastScalarVolumeInputSpec
    output_spec = CastScalarVolumeOutputSpec
    _cmd = 'CastScalarVolume '
    _outputs_filenames = {'OutputVolume': 'OutputVolume.nii'}
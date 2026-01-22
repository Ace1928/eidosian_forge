from nipype.interfaces.base import (
import os
class SubtractScalarVolumes(SEMLikeCommandLine):
    """title: Subtract Scalar Volumes

    category: Filtering.Arithmetic

    description: Subtracts two images. Although all image types are supported on input, only signed types are produced. The two images do not have to have the same dimensions.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/Subtract

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = SubtractScalarVolumesInputSpec
    output_spec = SubtractScalarVolumesOutputSpec
    _cmd = 'SubtractScalarVolumes '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
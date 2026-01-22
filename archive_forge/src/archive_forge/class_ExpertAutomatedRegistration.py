from nipype.interfaces.base import (
import os
class ExpertAutomatedRegistration(SEMLikeCommandLine):
    """title: Expert Automated Registration

    category: Legacy.Registration

    description: Provides rigid, affine, and BSpline registration methods via a simple GUI

    version: 0.1.0.$Revision: 2104 $(alpha)

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ExpertAutomatedRegistration

    contributor: Stephen R Aylward (Kitware), Casey B Goodlett (Kitware)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = ExpertAutomatedRegistrationInputSpec
    output_spec = ExpertAutomatedRegistrationOutputSpec
    _cmd = 'ExpertAutomatedRegistration '
    _outputs_filenames = {'resampledImage': 'resampledImage.nii', 'saveTransform': 'saveTransform.txt'}
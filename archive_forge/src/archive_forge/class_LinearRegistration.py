from nipype.interfaces.base import (
import os
class LinearRegistration(SEMLikeCommandLine):
    """title: Linear Registration

    category: Legacy.Registration

    description: Registers two images together using a rigid transform and mutual information.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/LinearRegistration

    contributor: Daniel Blezek (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = LinearRegistrationInputSpec
    output_spec = LinearRegistrationOutputSpec
    _cmd = 'LinearRegistration '
    _outputs_filenames = {'resampledmovingfilename': 'resampledmovingfilename.nii', 'outputtransform': 'outputtransform.txt'}
from nipype.interfaces.base import (
import os
class BSplineDeformableRegistration(SEMLikeCommandLine):
    """title: BSpline Deformable Registration

    category: Legacy.Registration

    description: Registers two images together using BSpline transform and mutual information.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/BSplineDeformableRegistration

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = BSplineDeformableRegistrationInputSpec
    output_spec = BSplineDeformableRegistrationOutputSpec
    _cmd = 'BSplineDeformableRegistration '
    _outputs_filenames = {'resampledmovingfilename': 'resampledmovingfilename.nii', 'outputtransform': 'outputtransform.txt', 'outputwarp': 'outputwarp.nrrd'}
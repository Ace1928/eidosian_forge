import os
from ...base import (
class BRAINSTransformFromFiducials(SEMLikeCommandLine):
    """title: Fiducial Registration (BRAINS)

    category: Registration.Specialized

    description: Computes a rigid, similarity or affine transform from a matched list of fiducials

    version: 0.1.0.$Revision$

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Modules:TransformFromFiducials-Documentation-3.6

    contributor: Casey B Goodlett

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = BRAINSTransformFromFiducialsInputSpec
    output_spec = BRAINSTransformFromFiducialsOutputSpec
    _cmd = ' BRAINSTransformFromFiducials '
    _outputs_filenames = {'saveTransform': 'saveTransform.h5'}
    _redirect_x = False
from nipype.interfaces.base import (
import os
class LabelMapSmoothing(SEMLikeCommandLine):
    """title: Label Map Smoothing

    category: Surface Models

    description: This filter smoothes a binary label map.  With a label map as input, this filter runs an anti-alising algorithm followed by a Gaussian smoothing algorithm.  The output is a smoothed label map.

    version: 1.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/LabelMapSmoothing

    contributor: Dirk Padfield (GE), Josh Cates (Utah), Ross Whitaker (Utah)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.  This filter is based on work developed at the University of Utah, and implemented at GE Research.
    """
    input_spec = LabelMapSmoothingInputSpec
    output_spec = LabelMapSmoothingOutputSpec
    _cmd = 'LabelMapSmoothing '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
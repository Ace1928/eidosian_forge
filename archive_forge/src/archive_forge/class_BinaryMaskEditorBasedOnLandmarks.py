import os
from ...base import (
class BinaryMaskEditorBasedOnLandmarks(SEMLikeCommandLine):
    """title: BRAINS Binary Mask Editor Based On Landmarks(BRAINS)

    category: Segmentation.Specialized

    version: 1.0

    documentation-url: http://www.nitrc.org/projects/brainscdetector/
    """
    input_spec = BinaryMaskEditorBasedOnLandmarksInputSpec
    output_spec = BinaryMaskEditorBasedOnLandmarksOutputSpec
    _cmd = ' BinaryMaskEditorBasedOnLandmarks '
    _outputs_filenames = {'outputBinaryVolume': 'outputBinaryVolume.nii'}
    _redirect_x = False
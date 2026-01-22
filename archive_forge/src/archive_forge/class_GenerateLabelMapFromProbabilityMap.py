import os
from ...base import (
class GenerateLabelMapFromProbabilityMap(SEMLikeCommandLine):
    """title: Label Map from Probability Images

    category: Utilities.BRAINS

    description: Given a list of probability maps for labels, create a discrete label map where only the highest probability region is used for the labeling.

    version: 0.1

    contributor: University of Iowa Department of Psychiatry, http:://www.psychiatry.uiowa.edu
    """
    input_spec = GenerateLabelMapFromProbabilityMapInputSpec
    output_spec = GenerateLabelMapFromProbabilityMapOutputSpec
    _cmd = ' GenerateLabelMapFromProbabilityMap '
    _outputs_filenames = {'outputLabelVolume': 'outputLabelVolume.nii.gz'}
    _redirect_x = False
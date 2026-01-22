import os
from ...base import (
class CleanUpOverlapLabels(SEMLikeCommandLine):
    """title: Clean Up Overla Labels

    category: Utilities.BRAINS

    description: Take a series of input binary images and clean up for those overlapped area. Binary volumes given first always wins out

    version: 0.1.0

    contributor: Eun Young Kim
    """
    input_spec = CleanUpOverlapLabelsInputSpec
    output_spec = CleanUpOverlapLabelsOutputSpec
    _cmd = ' CleanUpOverlapLabels '
    _outputs_filenames = {'outputBinaryVolumes': 'outputBinaryVolumes.nii'}
    _redirect_x = False
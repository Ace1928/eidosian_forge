import os
from ...base import (
class BRAINSCreateLabelMapFromProbabilityMaps(SEMLikeCommandLine):
    """title: Create Label Map From Probability Maps (BRAINS)

    category: Segmentation.Specialized

    description: Given A list of Probability Maps, generate a LabelMap.
    """
    input_spec = BRAINSCreateLabelMapFromProbabilityMapsInputSpec
    output_spec = BRAINSCreateLabelMapFromProbabilityMapsOutputSpec
    _cmd = ' BRAINSCreateLabelMapFromProbabilityMaps '
    _outputs_filenames = {'dirtyLabelVolume': 'dirtyLabelVolume.nii', 'cleanLabelVolume': 'cleanLabelVolume.nii'}
    _redirect_x = False
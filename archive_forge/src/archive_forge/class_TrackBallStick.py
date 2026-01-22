import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackBallStick(Track):
    """
    Performs streamline tractography using ball-stick fitted data

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> track = cmon.TrackBallStick()
    >>> track.inputs.in_file = 'ballstickfit_data.Bfloat'
    >>> track.inputs.seed_file = 'seed_mask.nii'
    >>> track.run()                  # doctest: +SKIP

    """

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'ballstick'
        return super(TrackBallStick, self).__init__(command, **inputs)
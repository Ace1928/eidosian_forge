import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class SphericallyDeconvolutedStreamlineTrack(StreamlineTrack):
    """
    Performs streamline tracking using spherically deconvolved data

    Specialized interface to StreamlineTrack. This interface is used for
    streamline tracking from spherically deconvolved data, and calls
    the MRtrix function 'streamtrack' with the option 'SD_STREAM'

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> sdtrack = mrt.SphericallyDeconvolutedStreamlineTrack()
    >>> sdtrack.inputs.in_file = 'data.Bfloat'
    >>> sdtrack.inputs.seed_file = 'seed_mask.nii'
    >>> sdtrack.run()                                          # doctest: +SKIP
    """
    input_spec = StreamlineTrackInputSpec

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'SD_STREAM'
        return super(SphericallyDeconvolutedStreamlineTrack, self).__init__(command, **inputs)
import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class ProbabilisticSphericallyDeconvolutedStreamlineTrackInputSpec(StreamlineTrackInputSpec):
    maximum_number_of_trials = traits.Int(argstr='-trials %s', desc='Set the maximum number of sampling trials at each point (only used for probabilistic tracking).')
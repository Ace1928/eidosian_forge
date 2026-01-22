import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class StreamlineTrack(CommandLine):
    """
    Performs tractography using one of the following models:
    'dt_prob', 'dt_stream', 'sd_prob', 'sd_stream',
    Where 'dt' stands for diffusion tensor, 'sd' stands for spherical
    deconvolution, and 'prob' stands for probabilistic.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> strack = mrt.StreamlineTrack()
    >>> strack.inputs.inputmodel = 'SD_PROB'
    >>> strack.inputs.in_file = 'data.Bfloat'
    >>> strack.inputs.seed_file = 'seed_mask.nii'
    >>> strack.inputs.mask_file = 'mask.nii'
    >>> strack.cmdline
    'streamtrack -mask mask.nii -seed seed_mask.nii SD_PROB data.Bfloat data_tracked.tck'
    >>> strack.run()                                    # doctest: +SKIP
    """
    _cmd = 'streamtrack'
    input_spec = StreamlineTrackInputSpec
    output_spec = StreamlineTrackOutputSpec
import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class FilterTracks(CommandLine):
    """
    Use regions-of-interest to select a subset of tracks
    from a given MRtrix track file.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> filt = mrt.FilterTracks()
    >>> filt.inputs.in_file = 'tracks.tck'
    >>> filt.run()                                 # doctest: +SKIP
    """
    _cmd = 'filter_tracks'
    input_spec = FilterTracksInputSpec
    output_spec = FilterTracksOutputSpec
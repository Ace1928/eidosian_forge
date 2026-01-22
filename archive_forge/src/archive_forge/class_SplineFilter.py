import os
from ..base import (
class SplineFilter(CommandLine):
    """
    Smoothes TrackVis track files with a B-Spline filter.

    Helps remove redundant track points and segments
    (thus reducing the size of the track file) and also
    make tracks nicely smoothed. It will NOT change the
    quality of the tracks or lose any original information.

    Example
    -------

    >>> import nipype.interfaces.diffusion_toolkit as dtk
    >>> filt = dtk.SplineFilter()
    >>> filt.inputs.track_file = 'tracks.trk'
    >>> filt.inputs.step_length = 0.5
    >>> filt.run()                                 # doctest: +SKIP
    """
    input_spec = SplineFilterInputSpec
    output_spec = SplineFilterOutputSpec
    _cmd = 'spline_filter'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['smoothed_track_file'] = os.path.abspath(self.inputs.output_file)
        return outputs
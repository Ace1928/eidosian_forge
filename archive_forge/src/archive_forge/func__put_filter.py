from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
@classmethod
def _put_filter(cls, filt, band):
    """
        Puts a filter in the band, internal helper function.

        Parameters
        ----------
        filt : :class:`Filter` instance
            Filter to be put into the band.
        band : numpy array
            Band in which the filter should be put.

        Notes
        -----
        The `band` must be an existing numpy array where the filter `filt` is
        put in, given the position of the filter. Out of range filters are
        truncated. If there are non-zero values in the filter band at the
        respective positions, the maximum value of the `band` and the filter
        `filt` is used.

        """
    if not isinstance(filt, Filter):
        raise ValueError('unable to determine start position of Filter')
    start = filt.start
    stop = start + len(filt)
    if start < 0:
        filt = filt[-start:]
        start = 0
    if stop > len(band):
        filt = filt[:-(stop - len(band))]
        stop = len(band)
    filter_position = band[start:stop]
    np.maximum(filt, filter_position, out=filter_position)
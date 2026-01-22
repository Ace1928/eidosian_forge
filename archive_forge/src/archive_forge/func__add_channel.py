from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
def _add_channel(notes, channel=0):
    """
    Adds a default channel to the notes if missing.

    Parameters
    ----------
    notes : numpy array, shape (num_notes, 2)
        Notes, one per row (column definition see notes).
    channel : int, optional
        Note channel if not defined by `notes`.

    Returns
    -------
    numpy array
        Notes (including note channel).

    Notes
    -----
    The note columns format must be (channel being optional):

    'onset' 'pitch' 'duration' 'velocity' ['channel']

    """
    if not notes.ndim == 2:
        raise ValueError('unknown format for `notes`')
    rows, columns = notes.shape
    if columns == 5:
        return notes
    elif columns == 4:
        channels = np.ones((rows, 1)) * channel
        return np.hstack((notes, channels))
    raise ValueError('unable to handle `notes` with %d columns' % columns)
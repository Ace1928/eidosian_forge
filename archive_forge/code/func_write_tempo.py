from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def write_tempo(tempi, filename, delimiter='\t', header=None, mirex=None):
    """
    Write the most dominant tempi and the relative strength to a file.

    Parameters
    ----------
    tempi : numpy array
        Array with the detected tempi (first column) and their strengths
        (second column).
    filename : str or file handle
        Output file.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.
    mirex : bool, deprecated
        Report the lower tempo first (as required by MIREX).

    Returns
    -------
    tempo_1 : float
        The most dominant tempo.
    tempo_2 : float
        The second most dominant tempo.
    strength : float
        Their relative strength.

    """
    tempi = np.array(tempi, ndmin=2)
    t1 = t2 = strength = np.nan
    if len(tempi) == 1:
        t1 = tempi[0][0]
        strength = 1.0
    elif len(tempi) > 1:
        t1, t2 = tempi[:2, 0]
        strength = tempi[0, 1] / sum(tempi[:2, 1])
    if mirex is not None:
        import warnings
        warnings.warn('`mirex` argument is deprecated as of version 0.16 and will be removed in version 0.17. Please sort the tempi manually')
        if t1 > t2:
            t1, t2, strength = (t2, t1, 1.0 - strength)
    out = np.array([t1, t2, strength], ndmin=2)
    write_events(out, filename, fmt=['%.2f', '%.2f', '%.2f'], delimiter=delimiter, header=header)
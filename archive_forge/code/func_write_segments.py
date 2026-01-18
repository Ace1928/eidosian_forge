from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def write_segments(segments, filename, fmt=None, delimiter='\t', header=None):
    """
    Write labelled segments to a file.

    Parameters
    ----------
    segments : numpy structured array
        Labelled segments, one per row (column definition see SEGMENT_DTYPE).
    filename : str or file handle
        Output filename or handle.
    fmt : str or sequence of strs, optional
        A sequence of formats (e.g. ['%.3f', '%.3f', '%s']), or a multi-format
        string (e.g. '%.3f %.3f %s'), in which case `delimiter` is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

    Returns
    -------
    numpy structured array
        Labelled segments

    Notes
    -----
    Labelled segments are represented as numpy structured array with three
    named columns: 'start' contains the start position (e.g. seconds),
    'end' the end position, and 'label' the segment label.

    """
    if fmt is None:
        fmt = ['%.3f', '%.3f', '%s']
    write_events(segments, filename, fmt=fmt, delimiter=delimiter, header=header)
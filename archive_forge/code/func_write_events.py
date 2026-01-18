from __future__ import absolute_import, division, print_function
import io as _io
import contextlib
import numpy as np
from .audio import load_audio_file
from .midi import load_midi, write_midi
from ..utils import suppress_warnings, string_types
def write_events(events, filename, fmt='%.3f', delimiter='\t', header=None):
    """
    Write the events to a file, one event per line.

    Parameters
    ----------
    events : numpy array
        Events to be written to file.
    filename : str or file handle
        File to write the events to.
    fmt : str or sequence of strs, optional
        A single format (e.g. '%.3f'), a sequence of formats, or a multi-format
        string (e.g. '%.3f %.3f'), in which case `delimiter` is ignored.
    delimiter : str, optional
        String or character separating columns.
    header : str, optional
        String that will be written at the beginning of the file as comment.

    """
    events = np.array(events)
    if isinstance(fmt, (list, tuple)):
        fmt = delimiter.join(fmt)
    with open_file(filename, 'wb') as f:
        if header is not None:
            f.write(bytes(('# ' + header + '\n').encode(ENCODING)))
        for e in events:
            try:
                string = fmt % tuple(e.tolist())
            except AttributeError:
                string = e
            except TypeError:
                string = fmt % e
            f.write(bytes((string + '\n').encode(ENCODING)))
            f.flush()
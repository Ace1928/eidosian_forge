from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
from . import beats, chords, downbeats, key, notes, onsets, tempo
from .beats import (BeatDetectionProcessor, BeatTrackingProcessor,
from .chords import (CNNChordFeatureProcessor, CRFChordRecognitionProcessor,
from .downbeats import (RNNDownBeatProcessor, DBNDownBeatTrackingProcessor,
from .key import CNNKeyRecognitionProcessor
from .notes import RNNPianoNoteProcessor, NotePeakPickingProcessor
from .onsets import (CNNOnsetProcessor, OnsetPeakPickingProcessor,
from .tempo import TempoEstimationProcessor
class Activations(np.ndarray):
    """
    The Activations class extends a numpy ndarray with a frame rate (fps)
    attribute.

    Parameters
    ----------
    data : str, file handle or numpy array
        Either file name/handle to read the data from or array.
    fps : float, optional
        Frames per second (must be set if `data` is given as an array).
    sep : str, optional
        Separator between activation values (if read from file).
    dtype : numpy dtype
        Data-type the activations are stored/saved/kept.

    Attributes
    ----------
    fps : float
        Frames per second.

    Notes
    -----
    If a filename or file handle is given, an undefined or empty separator
    means that the file should be treated as a numpy binary file.
    Only binary files can store the frame rate of the activations.
    Text files should not be used for anything else but manual inspection
    or I/O with other programs.

    """

    def __init__(self, data, fps=None, sep=None, dtype=np.float32):
        pass

    def __new__(cls, data, fps=None, sep=None, dtype=np.float32):
        import io
        if isinstance(data, np.ndarray):
            obj = np.asarray(data, dtype=dtype).view(cls)
            obj.fps = fps
        elif isinstance(data, (str, io.IOBase)):
            obj = cls.load(data, fps, sep)
        else:
            raise TypeError('wrong input data for Activations')
        if obj.fps is None:
            raise TypeError('frame rate for Activations must be set')
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.fps = getattr(obj, 'fps', None)

    @classmethod
    def load(cls, infile, fps=None, sep=None):
        """
        Load the activations from a file.

        Parameters
        ----------
        infile : str or file handle
            Input file name or file handle.
        fps : float, optional
            Frames per second; if set, it overwrites the saved frame rate.
        sep : str, optional
            Separator between activation values.

        Returns
        -------
        :class:`Activations` instance
            :class:`Activations` instance.

        Notes
        -----
        An undefined or empty separator means that the file should be treated
        as a numpy binary file.
        Only binary files can store the frame rate of the activations.
        Text files should not be used for anything else but manual inspection
        or I/O with other programs.

        """
        if sep in [None, '']:
            data = np.load(infile)
            if isinstance(data, np.lib.npyio.NpzFile):
                if fps is None:
                    fps = float(data['fps'])
                data = data['activations']
        else:
            data = np.loadtxt(infile, delimiter=sep)
        if data.ndim > 1 and data.shape[1] == 1:
            data = data.flatten()
        return cls(data, fps)

    def save(self, outfile, sep=None, fmt='%.5f'):
        """
        Save the activations to a file.

        Parameters
        ----------
        outfile : str or file handle
            Output file name or file handle.
        sep : str, optional
            Separator between activation values if saved as text file.
        fmt : str, optional
            Format of the values if saved as text file.

        Notes
        -----
        An undefined or empty separator means that the file should be treated
        as a numpy binary file.
        Only binary files can store the frame rate of the activations.
        Text files should not be used for anything else but manual inspection
        or I/O with other programs.

        If the activations are a 1D array, its values are interpreted as
        features of a single time step, i.e. all values are printed in a single
        line. If you want each value to appear in an individual line, use '\\n'
        as a separator.

        If the activations are a 2D array, the first axis corresponds to the
        time dimension, i.e. the features are separated by `sep` and the time
        steps are printed in separate lines. If you like to swap the
        dimensions, please use the `T` attribute.

        """
        if sep in [None, '']:
            npz = {'activations': self, 'fps': self.fps}
            np.savez(outfile, **npz)
        else:
            if self.ndim > 2:
                raise ValueError('Only 1D and 2D activations can be saved in human readable text format.')
            header = 'FPS:%f' % self.fps
            np.savetxt(outfile, np.atleast_2d(self), fmt=fmt, delimiter=sep, header=header)
        outfile.close()
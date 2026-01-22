from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
class LoadBeatsProcessor(Processor):
    """
    Load beat times from file or handle.

    """

    def __init__(self, beats, files=None, beats_suffix=None, **kwargs):
        from ..utils import search_files
        if isinstance(files, list) and beats_suffix is not None:
            beats = search_files(files, suffix=beats_suffix)
            self.mode = 'batch'
        else:
            self.mode = 'single'
        self.beats = beats
        self.beats_suffix = beats_suffix

    def process(self, data=None, **kwargs):
        """
        Load the beats from file (handle) or read them from STDIN.

        """
        if self.mode == 'single':
            return self.process_single()
        elif self.mode == 'batch':
            return self.process_batch(data)
        else:
            raise ValueError("don't know how to obtain the beats")

    def process_single(self):
        """
        Load the beats in bulk-mode (i.e. all at once) from the input stream
        or file.

        Returns
        -------
        beats : numpy array
            Beat positions [seconds].

        """
        from ..io import load_events
        return load_events(self.beats)

    def process_batch(self, filename):
        """
        Load beat times from file.

        First match the given input filename to the beat filenames, then load
        the beats.

        Parameters
        ----------
        filename : str
            Input file name.

        Returns
        -------
        beats : numpy array
            Beat positions [seconds].

        Notes
        -----
        Both the file names to search for the beats as well as the suffix to
        determine the beat files must be given at instantiation time.

        """
        import os
        from ..utils import match_file
        if not isinstance(filename, string_types):
            raise SystemExit('Please supply a filename, not %s.' % filename)
        basename, ext = os.path.splitext(os.path.basename(filename))
        matches = match_file(basename, self.beats, suffix=ext, match_suffix=self.beats_suffix)
        if not matches:
            raise SystemExit("can't find a beat file for %s" % filename)
        beats = np.loadtxt(matches[0])
        if beats.ndim == 2:
            beats = beats[:, 0]
        return beats

    @staticmethod
    def add_arguments(parser, beats=sys.stdin, beats_suffix='.beats.txt'):
        """
        Add beat loading related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats : FileType, optional
            Where to read the beats from ('single' mode).
        beats_suffix : str, optional
            Suffix of beat files ('batch' mode)

        Returns
        -------
        argparse argument group
            Beat loading argument parser group.

        """
        import argparse
        g = parser.add_argument_group('beat loading arguments')
        g.add_argument('--beats', type=argparse.FileType('rb'), default=beats, help='where/how to read the beat positions from [default: single: STDIN]')
        g.add_argument('--beats_suffix', type=str, default=beats_suffix, help='file suffix of the beat files [default: %(default)s]')
        return g
from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
class Evaluation(SimpleEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure based on
    numpy arrays or lists with true/false positive/negative detections.

    Parameters
    ----------
    tp : list or numpy array
        True positive detections.
    fp : list or numpy array
        False positive detections.
    tn : list or numpy array
        True negative detections.
    fn : list or numpy array
        False negative detections.
    name : str
        Name to be displayed.

    """

    def __init__(self, tp=None, fp=None, tn=None, fn=None, **kwargs):
        if tp is None:
            tp = []
        if fp is None:
            fp = []
        if tn is None:
            tn = []
        if fn is None:
            fn = []
        super(Evaluation, self).__init__(**kwargs)
        self.tp = np.asarray(list(tp), dtype=np.float)
        self.fp = np.asarray(list(fp), dtype=np.float)
        self.tn = np.asarray(list(tn), dtype=np.float)
        self.fn = np.asarray(list(fn), dtype=np.float)

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return len(self.tp)

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return len(self.fp)

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return len(self.tn)

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return len(self.fn)
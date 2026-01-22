from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
class BeatEvaluation(OnsetEvaluation):
    """
    Beat evaluation class.

    Parameters
    ----------
    detections : str, list or numpy array
        Detected beats.
    annotations : str, list or numpy array
        Annotated ground truth beats.
    fmeasure_window : float, optional
        F-measure evaluation window [seconds]
    pscore_tolerance : float, optional
        P-Score tolerance [fraction of the median beat interval].
    cemgil_sigma : float, optional
        Sigma of Gaussian window for Cemgil accuracy.
    goto_threshold : float, optional
        Threshold for Goto error.
    goto_sigma : float, optional
        Sigma for Goto error.
    goto_mu : float, optional
        Mu for Goto error.
    continuity_phase_tolerance : float, optional
        Continuity phase tolerance.
    continuity_tempo_tolerance : float, optional
        Ccontinuity tempo tolerance.
    information_gain_bins : int, optional
        Number of bins for for the information gain beat error histogram.
    offbeat : bool, optional
        Include offbeat variation.
    double : bool, optional
        Include double and half tempo variations (and offbeat thereof).
    triple : bool, optional
        Include triple and third tempo variations (and offbeats thereof).
    skip : float, optional
        Skip the first `skip` seconds for evaluation.
    downbeats : bool, optional
        Evaluate downbeats instead of beats.

    Notes
    -----
    The `offbeat`, `double`, and `triple` variations of the beat sequences are
    used only for AMLc/AMLt.

    """
    METRIC_NAMES = [('fmeasure', 'F-measure'), ('pscore', 'P-score'), ('cemgil', 'Cemgil'), ('goto', 'Goto'), ('cmlc', 'CMLc'), ('cmlt', 'CMLt'), ('amlc', 'AMLc'), ('amlt', 'AMLt'), ('information_gain', 'D'), ('global_information_gain', 'Dg')]

    def __init__(self, detections, annotations, fmeasure_window=FMEASURE_WINDOW, pscore_tolerance=PSCORE_TOLERANCE, cemgil_sigma=CEMGIL_SIGMA, goto_threshold=GOTO_THRESHOLD, goto_sigma=GOTO_SIGMA, goto_mu=GOTO_MU, continuity_phase_tolerance=CONTINUITY_PHASE_TOLERANCE, continuity_tempo_tolerance=CONTINUITY_TEMPO_TOLERANCE, information_gain_bins=INFORMATION_GAIN_BINS, offbeat=True, double=True, triple=True, skip=0, downbeats=False, **kwargs):
        detections = np.array(detections, dtype=np.float, ndmin=1)
        annotations = np.array(annotations, dtype=np.float, ndmin=1)
        if detections.ndim > 1:
            if downbeats:
                detections = detections[detections[:, 1] == 1][:, 0]
            else:
                detections = detections[:, 0]
        if annotations.ndim > 1:
            if downbeats:
                annotations = annotations[annotations[:, 1] == 1][:, 0]
            else:
                annotations = annotations[:, 0]
        detections = np.sort(detections)
        annotations = np.sort(annotations)
        if skip > 0:
            start_idx = np.searchsorted(detections, skip, 'right')
            detections = detections[start_idx:]
            start_idx = np.searchsorted(annotations, skip, 'right')
            annotations = annotations[start_idx:]
        super(BeatEvaluation, self).__init__(detections, annotations, window=fmeasure_window, **kwargs)
        self.pscore = pscore(detections, annotations, pscore_tolerance)
        self.cemgil = cemgil(detections, annotations, cemgil_sigma)
        self.goto = goto(detections, annotations, goto_threshold, goto_sigma, goto_mu)
        scores = continuity(detections, annotations, continuity_tempo_tolerance, continuity_phase_tolerance, offbeat, double, triple)
        self.cmlc, self.cmlt, self.amlc, self.amlt = scores
        scores = information_gain(detections, annotations, information_gain_bins)
        self.information_gain, self.error_histogram = scores

    @property
    def global_information_gain(self):
        """Global information gain."""
        return self.information_gain

    def tostring(self, **kwargs):
        return tostring(self)
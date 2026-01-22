from __future__ import absolute_import, division, print_function
import numpy as np
from . import Evaluation, MeanEvaluation, SumEvaluation, evaluation_io
from ..io import load_onsets
from ..utils import combine_events
class OnsetSumEvaluation(SumEvaluation, OnsetEvaluation):
    """
    Class for summing onset evaluations.

    """

    @property
    def errors(self):
        """Errors of the true positive detections wrt. the ground truth."""
        if not self.eval_objects:
            return np.zeros(0)
        return np.concatenate([e.errors for e in self.eval_objects])
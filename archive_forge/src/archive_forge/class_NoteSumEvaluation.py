from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import (evaluation_io, MultiClassEvaluation, SumEvaluation,
from .onsets import onset_evaluation, OnsetEvaluation
from ..io import load_notes
class NoteSumEvaluation(SumEvaluation, NoteEvaluation):
    """
    Class for summing note evaluations.

    """

    @property
    def errors(self):
        """Errors of the true positive detections wrt. the ground truth."""
        if not self.eval_objects:
            return np.zeros((0, 2))
        return np.concatenate([e.errors for e in self.eval_objects])
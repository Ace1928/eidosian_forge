from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
def _process_dbn(process_tuple):
    """
    Extract the best path through the state space in an observation sequence.

    This proxy function is necessary to process different sequences in parallel
    using the multiprocessing module.

    Parameters
    ----------
    process_tuple : tuple
        Tuple with (HMM, observations).

    Returns
    -------
    path : numpy array
        Best path through the state space.
    log_prob : float
        Log probability of the path.

    """
    return process_tuple[0].viterbi(process_tuple[1])
from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types

        Add DBN related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        beats_per_bar : int or list, optional
            Number of beats per bar to be modeled. Can be either a single
            number or a list with bar lengths (in beats).
        observation_weight : float, optional
            Weight for the activations at downbeat times.
        meter_change_prob : float, optional
            Probability to change meter at bar boundaries.

        Returns
        -------
        parser_group : argparse argument group
            DBN bar tracking argument parser group

        
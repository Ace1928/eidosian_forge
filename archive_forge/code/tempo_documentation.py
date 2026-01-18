from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor

        Add tempo estimation related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.
        method : {'comb', 'acf', 'dbn'}
            Method used for tempo estimation.
        min_bpm : float, optional
            Minimum tempo to detect [bpm].
        max_bpm : float, optional
            Maximum tempo to detect [bpm].
        act_smooth : float, optional
            Smooth the activation function over `act_smooth` seconds.
        hist_smooth : int, optional
            Smooth the tempo histogram over `hist_smooth` bins.
        hist_buffer : float, optional
            Aggregate the tempo histogram over `hist_buffer` seconds.
        alpha : float, optional
            Scaling factor for the comb filter.

        Returns
        -------
        parser_group : argparse argument group
            Tempo argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        
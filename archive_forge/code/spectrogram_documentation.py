from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,

        Return the a multi-band representation of the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to :class:`MultiBandSpectrogram`.

        Returns
        -------
        multi_band_spec : :class:`MultiBandSpectrogram` instance
            Spectrogram split into multiple bands.

        
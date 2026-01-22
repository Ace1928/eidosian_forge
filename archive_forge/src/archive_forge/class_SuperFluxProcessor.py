from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class SuperFluxProcessor(SequentialProcessor):
    """
    Spectrogram processor which sets the default values suitable for the
    SuperFlux algorithm.

    """

    def __init__(self, **kwargs):
        from .stft import ShortTimeFourierTransformProcessor
        filterbank = kwargs.pop('filterbank', FILTERBANK)
        num_bands = kwargs.pop('num_bands', 24)
        norm_filters = kwargs.pop('norm_filters', False)
        diff_ratio = kwargs.pop('diff_ratio', 0.5)
        diff_max_bins = kwargs.pop('diff_max_bins', 3)
        positive_diffs = kwargs.pop('positive_diffs', True)
        stft = ShortTimeFourierTransformProcessor(**kwargs)
        spec = SpectrogramProcessor(**kwargs)
        filt = FilteredSpectrogramProcessor(filterbank=filterbank, num_bands=num_bands, norm_filters=norm_filters, **kwargs)
        log = LogarithmicSpectrogramProcessor(**kwargs)
        diff = SpectrogramDifferenceProcessor(diff_ratio=diff_ratio, diff_max_bins=diff_max_bins, positive_diffs=positive_diffs, **kwargs)
        super(SuperFluxProcessor, self).__init__([stft, spec, filt, log, diff])
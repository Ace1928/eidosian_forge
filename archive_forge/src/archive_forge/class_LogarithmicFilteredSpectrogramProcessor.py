from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class LogarithmicFilteredSpectrogramProcessor(Processor):
    """
    Logarithmic Filtered Spectrogram Processor class.

    Parameters
    ----------
    filterbank : :class:`.audio.filters.Filterbank`
        Filterbank used to filter a spectrogram.
    num_bands : int
        Number of bands (per octave).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    mul : float, optional
        Multiply the magnitude spectrogram with this factor before taking the
        logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.

    """

    def __init__(self, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, mul=MUL, add=ADD, **kwargs):
        self.filterbank = filterbank
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.fref = fref
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters
        self.mul = mul
        self.add = add

    def process(self, data, **kwargs):
        """
        Perform filtering and logarithmic scaling of a spectrogram.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to
            :class:`LogarithmicFilteredSpectrogram`.

        Returns
        -------
        log_filt_spec : :class:`LogarithmicFilteredSpectrogram` instance
            Logarithmically scaled filtered spectrogram.

        """
        args = dict(filterbank=self.filterbank, num_bands=self.num_bands, fmin=self.fmin, fmax=self.fmax, fref=self.fref, norm_filters=self.norm_filters, unique_filters=self.unique_filters, mul=self.mul, add=self.add)
        args.update(kwargs)
        data = LogarithmicFilteredSpectrogram(data, **args)
        self.filterbank = data.filterbank
        return data
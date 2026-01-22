from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class FilteredSpectrogram(Spectrogram):
    """
    FilteredSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
    filterbank : :class:`.audio.filters.Filterbank`, optional
        Filterbank class or instance; if a class is given (rather than an
        instance), one will be created with the given type and parameters.
    num_bands : int, optional
        Number of filter bands (per octave, depending on the type of the
        `filterbank`).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter bands of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated
        with these additional keyword arguments.

    Examples
    --------
    Create a :class:`FilteredSpectrogram` from a :class:`Spectrogram` (or
    anything it can be instantiated from. Per default a
    :class:`.madmom.audio.filters.LogarithmicFilterbank` with 12 bands per
    octave is used.

    >>> spec = FilteredSpectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    FilteredSpectrogram([[ 5.66156, 6.30141, ..., 0.05426, 0.06461],
                         [ 8.44266, 8.69582, ..., 0.07703, 0.0902 ],
                         ...,
                         [10.04626, 1.12018, ..., 0.0487 , 0.04282],
                         [ 8.60186, 6.81195, ..., 0.03721, 0.03371]],
                        dtype=float32)

    The resulting spectrogram has fewer frequency bins, with the centers of
    the bins aligned logarithmically (lower frequency bins still have a linear
    spacing due to the coarse resolution of the DFT at low frequencies):

    >>> spec.shape
    (281, 81)
    >>> spec.num_bins
    81
    >>> spec.bin_frequencies  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([    43.06641,    64.59961,    86.13281,   107.66602,
              129.19922,   150.73242,   172.26562,   193.79883, ...,
            10551.26953, 11175.73242, 11843.26172, 12553.85742,
            13285.98633, 14082.71484, 14922.50977, 15805.37109])

    The filterbank used to filter the spectrogram is saved as an attribute:

    >>> spec.filterbank  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicFilterbank([[0., 0., ..., 0., 0.],
                           [0., 0., ..., 0., 0.],
                           ...,
                           [0., 0., ..., 0., 0.],
                           [0., 0., ..., 0., 0.]], dtype=float32)
    >>> spec.filterbank.num_bands
    81

    The filterbank can be chosen at instantiation time:

    >>> from madmom.audio.filters import MelFilterbank
    >>> spec = FilteredSpectrogram('tests/data/audio/sample.wav',     filterbank=MelFilterbank, num_bands=40)
    >>> type(spec.filterbank)
    <class 'madmom.audio.filters.MelFilterbank'>
    >>> spec.shape
    (281, 40)

    """

    def __init__(self, spectrogram, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, **kwargs):
        pass

    def __new__(cls, spectrogram, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        if inspect.isclass(filterbank) and issubclass(filterbank, Filterbank):
            filterbank = filterbank(spectrogram.bin_frequencies, num_bands=num_bands, fmin=fmin, fmax=fmax, fref=fref, norm_filters=norm_filters, unique_filters=unique_filters)
        if not isinstance(filterbank, Filterbank):
            raise TypeError('not a Filterbank type or instance: %s' % filterbank)
        data = np.dot(spectrogram, filterbank)
        obj = np.asarray(data).view(cls)
        obj.filterbank = filterbank
        obj.stft = spectrogram.stft
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.stft = getattr(obj, 'stft', None)
        self.filterbank = getattr(obj, 'filterbank', None)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.filterbank.center_frequencies
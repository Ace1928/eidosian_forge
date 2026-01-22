from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import scipy.fftpack as fftpack
from ..processors import Processor
from .signal import Signal, FramedSignal
class LocalGroupDelay(_PropertyMixin, np.ndarray):
    """
    Local Group Delay class.

    Parameters
    ----------
    stft : :class:`Phase` instance
         :class:`Phase` instance.
    kwargs : dict, optional
        If no :class:`Phase` instance was given, one is instantiated with
        these additional keyword arguments.

    Examples
    --------
    Create a :class:`LocalGroupDelay` from a :class:`ShortTimeFourierTransform`
    (or anything it can be instantiated from:

    >>> stft = ShortTimeFourierTransform('tests/data/audio/sample.wav')
    >>> lgd = LocalGroupDelay(stft)
    >>> lgd  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LocalGroupDelay([[-2.2851 , -2.25605, ...,  3.13525,  0. ],
                     [ 2.35804,  2.53786, ...,  1.76788,  0. ],
                     ...,
                     [-1.98..., -2.93039, ..., -1.77505,  0. ],
                     [ 2.7136 ,  2.60925, ...,  3.13318,  0. ]])


    """

    def __init__(self, phase, **kwargs):
        pass

    def __new__(cls, phase, **kwargs):
        if not isinstance(stft, Phase):
            phase = Phase(phase, circular_shift=True, **kwargs)
        if not phase.stft.circular_shift:
            warnings.warn("`circular_shift` of the STFT must be set to 'True' for correct local group delay")
        obj = np.asarray(local_group_delay(phase)).view(cls)
        obj.phase = phase
        obj.stft = phase.stft
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.phase = getattr(obj, 'phase', None)
        self.stft = getattr(obj, 'stft', None)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies
from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import scipy.fftpack as fftpack
from ..processors import Processor
from .signal import Signal, FramedSignal
class ShortTimeFourierTransformProcessor(Processor):
    """
    ShortTimeFourierTransformProcessor class.

    Parameters
    ----------
    window : numpy ufunc, optional
        Window function.
    fft_size : int, optional
        FFT size (should be a power of 2); if 'None', it is determined by the
        size of the frames; if is greater than the frame size, the frames are
        zero-padded accordingly.
    circular_shift : bool, optional
        Circular shift the individual frames before performing the FFT;
        needed for correct phase.
    include_nyquist : bool, optional
        Include the Nyquist frequency bin (sample rate / 2).

    Examples
    --------
    Create a :class:`ShortTimeFourierTransformProcessor` and call it with
    either a file name or a the output of a (Framed-)SignalProcessor to obtain
    a :class:`ShortTimeFourierTransform` instance.

    >>> proc = ShortTimeFourierTransformProcessor()
    >>> stft = proc('tests/data/audio/sample.wav')
    >>> stft  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ShortTimeFourierTransform([[-3.15249+0.j     ,  2.62216-3.02425j, ...,
                                -0.03634-0.00005j,  0.0367 +0.00029j],
                               [-4.28429+0.j     ,  2.02009+2.01264j, ...,
                                -0.01981-0.00933j, -0.00536+0.02162j],
                               ...,
                               [-4.92274+0.j     ,  4.09839-9.42525j, ...,
                                 0.0055 -0.00257j,  0.00137+0.00577j],
                               [-9.22709+0.j     ,  8.76929+4.0005j , ...,
                                 0.00981-0.00014j, -0.00984+0.00006j]],
                              dtype=complex64)

    """

    def __init__(self, window=np.hanning, fft_size=None, circular_shift=False, include_nyquist=False, **kwargs):
        self.window = window
        self.fft_size = fft_size
        self.circular_shift = circular_shift
        self.include_nyquist = include_nyquist
        self.fft_window = None
        self.fftw = None

    def process(self, data, **kwargs):
        """
        Perform FFT on a framed signal and return the STFT.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments passed to :class:`ShortTimeFourierTransform`.

        Returns
        -------
        stft : :class:`ShortTimeFourierTransform`
            :class:`ShortTimeFourierTransform` instance.

        """
        data = ShortTimeFourierTransform(data, window=self.window, fft_size=self.fft_size, circular_shift=self.circular_shift, include_nyquist=self.include_nyquist, fft_window=self.fft_window, fftw=self.fftw, **kwargs)
        self.fft_window = data.fft_window
        self.fftw = data.fftw
        return data

    @staticmethod
    def add_arguments(parser, window=None, fft_size=None):
        """
        Add STFT related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.
        window : numpy ufunc, optional
            Window function.
        fft_size : int, optional
            Use this size for FFT (should be a power of 2).

        Returns
        -------
        argparse argument group
            STFT argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        g = parser.add_argument_group('short-time Fourier transform arguments')
        if window is not None:
            g.add_argument('--window', dest='window', action='store', default=window, help='window function to use for FFT')
        if fft_size is not None:
            g.add_argument('--fft_size', action='store', type=int, default=fft_size, help='use this size for FFT (should be a power of 2) [default=%(default)i]')
        return g
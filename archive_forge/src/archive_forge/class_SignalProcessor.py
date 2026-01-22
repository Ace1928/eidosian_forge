from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
class SignalProcessor(Processor):
    """
    The :class:`SignalProcessor` class is a basic signal processor.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of the signal [Hz]; if set the signal will be re-sampled
        to that sample rate; if 'None' the sample rate of the audio file will
        be used.
    num_channels : int, optional
        Number of channels of the signal; if set, the signal will be reduced
        to that number of channels; if 'None' as many channels as present in
        the audio file are returned.
    start : float, optional
        Start position [seconds].
    stop : float, optional
        Stop position [seconds].
    norm : bool, optional
        Normalize the signal to the range [-1, +1].
    gain : float, optional
        Adjust the gain of the signal [dB].
    dtype : numpy data type, optional
        The data is returned with the given dtype. If 'None', it is returned
        with its original dtype, otherwise the signal gets rescaled. Integer
        dtypes use the complete value range, float dtypes the range [-1, +1].

    Examples
    --------
    Processor for loading the first two seconds of an audio file, re-sampling
    it to 22.05 kHz and down-mixing it to mono:

    >>> proc = SignalProcessor(sample_rate=22050, num_channels=1, stop=2)
    >>> sig = proc('tests/data/audio/sample.wav')
    >>> sig
    Signal([-2470, -2553, ...,  -173,  -265], dtype=int16)
    >>> sig.sample_rate
    22050
    >>> sig.num_channels
    1
    >>> sig.length
    2.0

    """

    def __init__(self, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, start=START, stop=STOP, norm=NORM, gain=GAIN, dtype=DTYPE, **kwargs):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.start = start
        self.stop = stop
        self.norm = norm
        self.gain = gain
        self.dtype = dtype

    def process(self, data, **kwargs):
        """
        Processes the given audio file.

        Parameters
        ----------
        data : numpy array, str or file handle
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments passed to :class:`Signal`.

        Returns
        -------
        signal : :class:`Signal` instance
            :class:`Signal` instance.

        """
        args = dict(sample_rate=self.sample_rate, num_channels=self.num_channels, start=self.start, stop=self.stop, norm=self.norm, gain=self.gain, dtype=self.dtype)
        args.update(kwargs)
        return Signal(data, **args)

    @staticmethod
    def add_arguments(parser, sample_rate=None, mono=None, start=None, stop=None, norm=None, gain=None):
        """
        Add signal processing related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        sample_rate : int, optional
            Re-sample the signal to this sample rate [Hz].
        mono : bool, optional
            Down-mix the signal to mono.
        start : float, optional
            Start position [seconds].
        stop : float, optional
            Stop position [seconds].
        norm : bool, optional
            Normalize the signal to the range [-1, +1].
        gain : float, optional
            Adjust the gain of the signal [dB].

        Returns
        -------
        argparse argument group
            Signal processing argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'. To
        include `start` and `stop` arguments with a default value of 'None',
        i.e. do not set any start or stop time, they can be set to 'True'.

        """
        g = parser.add_argument_group('signal processing arguments')
        if sample_rate is not None:
            g.add_argument('--sample_rate', action='store', type=int, default=sample_rate, help='re-sample the signal to this sample rate [Hz]')
        if mono is not None:
            g.add_argument('--mono', dest='num_channels', action='store_const', const=1, help='down-mix the signal to mono')
        if start is not None:
            g.add_argument('--start', action='store', type=float, help='start position of the signal [seconds]')
        if stop is not None:
            g.add_argument('--stop', action='store', type=float, help='stop position of the signal [seconds]')
        if norm is not None:
            g.add_argument('--norm', action='store_true', default=norm, help='normalize the signal [default=%(default)s]')
        if gain is not None:
            g.add_argument('--gain', action='store', type=float, default=gain, help='adjust the gain of the signal [dB, default=%(default).1f]')
        return g
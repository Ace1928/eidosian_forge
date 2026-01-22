from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
class FramedSignalProcessor(Processor):
    """
    Slice a Signal into frames.

    Parameters
    ----------
    frame_size : int, optional
        Size of one frame [samples].
    hop_size : float, optional
        Progress `hop_size` samples between adjacent frames.
    fps : float, optional
        Use given frames per second; if set, this computes and overwrites the
        given `hop_size` value.
    origin : int, optional
        Location of the window relative to the reference sample of a frame.
    end : int or str, optional
        End of signal handling (see :class:`FramedSignal`).
    num_frames : int, optional
        Number of frames to return.

    Notes
    -----
    When operating on live audio signals, `origin` must be set to 'stream' in
    order to retrieve always the last `frame_size` samples.

    Examples
    --------
    Processor for chopping a :class:`Signal` (or anything a :class:`Signal` can
    be instantiated from) into overlapping frames of size 2048, and a frame
    rate of 100 frames per second:

    >>> proc = FramedSignalProcessor(frame_size=2048, fps=100)
    >>> frames = proc('tests/data/audio/sample.wav')
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> frames[0]
    Signal([    0,     0, ..., -4666, -4589], dtype=int16)
    >>> frames[10]
    Signal([-6156, -5645, ...,  -253,   671], dtype=int16)
    >>> frames.hop_size
    441.0

    """

    def __init__(self, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fps=FPS, origin=ORIGIN, end=END_OF_SIGNAL, num_frames=NUM_FRAMES, **kwargs):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fps = fps
        self.origin = origin
        self.end = end
        self.num_frames = num_frames

    def process(self, data, **kwargs):
        """
        Slice the signal into (overlapping) frames.

        Parameters
        ----------
        data : :class:`Signal` instance
            Signal to be sliced into frames.
        kwargs : dict, optional
            Keyword arguments passed to :class:`FramedSignal`.

        Returns
        -------
        frames : :class:`FramedSignal` instance
            FramedSignal instance

        """
        args = dict(frame_size=self.frame_size, hop_size=self.hop_size, fps=self.fps, origin=self.origin, end=self.end, num_frames=self.num_frames)
        args.update(kwargs)
        if self.origin == 'stream':
            data = data[-self.frame_size:]
        return FramedSignal(data, **args)

    @staticmethod
    def add_arguments(parser, frame_size=FRAME_SIZE, fps=FPS, online=None):
        """
        Add signal framing related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        frame_size : int, optional
            Size of one frame in samples.
        fps : float, optional
            Frames per second.
        online : bool, optional
            Online mode (use only past signal information, i.e. align the
            window to the left of the reference sample).

        Returns
        -------
        argparse argument group
            Signal framing argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        g = parser.add_argument_group('signal framing arguments')
        if isinstance(frame_size, integer_types):
            g.add_argument('--frame_size', action='store', type=int, default=frame_size, help='frame size [samples, default=%(default)i]')
        elif isinstance(frame_size, list):
            from ..utils import OverrideDefaultListAction
            g.add_argument('--frame_size', type=int, default=frame_size, action=OverrideDefaultListAction, sep=',', help='(comma separated list of) frame size(s) to use [samples, default=%(default)s]')
        if fps is not None:
            g.add_argument('--fps', action='store', type=float, default=fps, help='frames per second [default=%(default).1f]')
        if online is False:
            g.add_argument('--online', dest='origin', action='store_const', const='online', default='offline', help='operate in online mode [default=offline]')
        elif online is True:
            g.add_argument('--offline', dest='origin', action='store_const', const='offline', default='online', help='operate in offline mode [default=online]')
        return g
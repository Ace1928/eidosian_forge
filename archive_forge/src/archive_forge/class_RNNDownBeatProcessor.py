from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
class RNNDownBeatProcessor(SequentialProcessor):
    """
    Processor to get a joint beat and downbeat activation function from
    multiple RNNs.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create a RNNDownBeatProcessor and pass a file through the processor.
    The returned 2d array represents the probabilities at each frame, sampled
    at 100 frames per second. The columns represent 'beat' and 'downbeat'.

    >>> proc = RNNDownBeatProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.RNNDownBeatProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[0.00011, 0.00037],
           [0.00008, 0.00043],
           ...,
           [0.00791, 0.00169],
           [0.03425, 0.00494]], dtype=float32)

    """

    def __init__(self, **kwargs):
        from functools import partial
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BLSTM
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        frame_sizes = [1024, 2048, 4096]
        num_bands = [3, 6, 12]
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM, **kwargs)
        act = partial(np.delete, obj=0, axis=1)
        super(RNNDownBeatProcessor, self).__init__((pre_processor, nn, act))
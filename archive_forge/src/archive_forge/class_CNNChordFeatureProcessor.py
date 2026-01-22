from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
class CNNChordFeatureProcessor(SequentialProcessor):
    """
    Extract learned features for chord recognition, as described in [1]_.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "A Fully Convolutional Deep Auditory Model for Musical Chord
           Recognition",
           Proceedings of IEEE International Workshop on Machine Learning for
           Signal Processing (MLSP), 2016.

    Examples
    --------
    >>> proc = CNNChordFeatureProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.chords.CNNChordFeatureProcessor object at 0x...>
    >>> features = proc('tests/data/audio/sample2.wav')
    >>> features.shape
    (41, 128)
    >>> features # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[0.05798, 0.     , ..., 0.02757, 0.014  ],
           [0.06604, 0.     , ..., 0.02898, 0.00886],
           ...,
           [0.00655, 0.1166 , ..., 0.00651, 0.     ],
           [0.01476, 0.11185, ..., 0.00287, 0.     ]])
    """

    def __init__(self, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
        from ..ml.nn import NeuralNetwork
        from ..models import CHORDS_CNN_FEAT
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=10)
        stft = ShortTimeFourierTransformProcessor()
        spec = LogarithmicFilteredSpectrogramProcessor(num_bands=24, fmin=60, fmax=2600, unique_filters=True)
        pad = _cnncfp_pad
        nn = NeuralNetwork.load(CHORDS_CNN_FEAT[0])
        superframes = _cnncfp_superframes
        avg = _cnncfp_avg
        super(CNNChordFeatureProcessor, self).__init__([sig, frames, stft, spec, pad, nn, superframes, avg])
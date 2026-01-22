import numpy as np
from ..processors import SequentialProcessor
class CNNKeyRecognitionProcessor(SequentialProcessor):
    """
    Recognise the global key of a musical piece using a Convolutional Neural
    Network as described in [1]_.

    Parameters
    ----------
    nn_files : list, optional
        List with trained CNN model files. Per default ('None'), an ensemble
        of networks will be used.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "Genre-Agnostic Key Classification with Convolutional Neural
           Networks", In Proceedings of the 19th International Society for
           Music Information Retrieval Conference (ISMIR), Paris, France, 2018.

    Examples
    --------
    Create a CNNKeyRecognitionProcessor and pass a file through it.
    The returned array represents the probability of each key class.

    >>> proc = CNNKeyRecognitionProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.key.CNNKeyRecognitionProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +NORMALIZE_WHITESPACE
    array([[0.03426, 0.0331 , 0.02979, 0.04423, 0.04215, 0.0311 , 0.05225,
            0.04263, 0.04141, 0.02907, 0.03755, 0.09546, 0.0431 , 0.02792,
            0.02138, 0.05589, 0.03276, 0.02786, 0.02415, 0.04608, 0.05329,
            0.02804, 0.03868, 0.08786]])
    """

    def __init__(self, nn_files=None, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..ml.nn.activations import softmax
        from ..models import KEY_CNN
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=5)
        stft = ShortTimeFourierTransformProcessor()
        spec = LogarithmicFilteredSpectrogramProcessor(num_bands=24, fmin=65, fmax=2100, unique_filters=True)
        nn_files = nn_files or KEY_CNN
        nn = NeuralNetworkEnsemble.load(nn_files)
        super(CNNKeyRecognitionProcessor, self).__init__([sig, frames, stft, spec, nn, add_axis, softmax])
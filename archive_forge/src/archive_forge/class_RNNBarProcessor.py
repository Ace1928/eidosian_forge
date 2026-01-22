from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
class RNNBarProcessor(Processor):
    """
    Retrieve a downbeat activation function from a signal and pre-determined
    beat positions by obtaining beat-synchronous harmonic and percussive
    features which are processed with a GRU-RNN.

    Parameters
    ----------
    beat_subdivisions : tuple, optional
        Number of beat subdivisions for the percussive and harmonic feature.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Downbeat Tracking Using Beat-Synchronous Features and Recurrent
           Networks",
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    Examples
    --------
    Create an RNNBarProcessor and pass an audio file and pre-determined (or
    given) beat positions through the processor. The returned tuple contains
    the beats positions and the probability to be a downbeat.

    >>> proc = RNNBarProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.RNNBarProcessor object at 0x...>
    >>> beats = np.loadtxt('tests/data/detections/sample.dbn_beat_tracker.txt')
    >>> downbeat_prob = proc(('tests/data/audio/sample.wav', beats))
    >>> np.around(downbeat_prob, decimals=3)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +NORMALIZE_ARRAYS
    array([[0.1  , 0.378],
           [0.45 , 0.19 ],
           [0.8  , 0.112],
           [1.12 , 0.328],
           [1.48 , 0.27 ],
           [1.8  , 0.181],
           [2.15 , 0.162],
           [2.49 ,   nan]])

    """

    def __init__(self, beat_subdivisions=(4, 2), fps=100, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor
        from ..audio.chroma import CLPChromaProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import DOWNBEATS_BGRU
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, fps=fps)
        stft = ShortTimeFourierTransformProcessor()
        spec = FilteredSpectrogramProcessor(num_bands=6, fmin=30.0, fmax=17000.0, norm_filters=True)
        log_spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True)
        self.perc_feat = SequentialProcessor((sig, frames, stft, spec, log_spec, diff))
        self.harm_feat = CLPChromaProcessor(fps=fps, fmin=27.5, fmax=4200.0, compression_factor=100, norm=True, threshold=0.001)
        self.perc_beat_sync = SyncronizeFeaturesProcessor(beat_subdivisions[0], fps=fps, **kwargs)
        self.harm_beat_sync = SyncronizeFeaturesProcessor(beat_subdivisions[1], fps=fps, **kwargs)
        self.perc_nn = NeuralNetworkEnsemble.load(DOWNBEATS_BGRU[0], **kwargs)
        self.harm_nn = NeuralNetworkEnsemble.load(DOWNBEATS_BGRU[1], **kwargs)

    def process(self, data, **kwargs):
        """
        Retrieve a downbeat activation function from a signal and beat
        positions.

        Parameters
        ----------
        data : tuple
            Tuple containg a signal or file (handle) and corresponding beat
            times [seconds].

        Returns
        -------
        numpy array, shape (num_beats, 2)
            Array containing the beat positions (first column) and the
            corresponding downbeat activations, i.e. the probability that a
            beat is a downbeat (second column).

        Notes
        -----
        Since features are synchronized to the beats, and the probability of
        being a downbeat depends on a whole beat duration, only num_beats-1
        activations can be computed and the last value is filled with 'NaN'.

        """
        signal, beats = data
        perc = self.perc_feat(signal)
        harm = self.harm_feat(signal)
        perc_synced = self.perc_beat_sync((perc, beats))
        harm_synced = self.harm_beat_sync((harm, beats))
        perc = self.perc_nn(perc_synced.reshape((len(perc_synced), -1)))
        harm = self.harm_nn(harm_synced.reshape((len(harm_synced), -1)))
        act = np.mean([perc, harm], axis=0)
        act = np.append(act, np.ones(1) * np.nan)
        return np.vstack((beats, act)).T
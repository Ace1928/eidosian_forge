import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def __pyin_helper(yin_frames, parabolic_shifts, sr, thresholds, boltzmann_parameter, beta_probs, no_trough_prob, min_period, fmin, n_pitch_bins, n_bins_per_semitone):
    yin_probs = np.zeros_like(yin_frames)
    for i, yin_frame in enumerate(yin_frames.T):
        is_trough = util.localmin(yin_frame)
        is_trough[0] = yin_frame[0] < yin_frame[1]
        trough_index, = np.nonzero(is_trough)
        if len(trough_index) == 0:
            continue
        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.less.outer(trough_heights, thresholds[1:])
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)
        trough_prior = scipy.stats.boltzmann.pmf(trough_positions, boltzmann_parameter, n_troughs)
        trough_prior[~trough_thresholds] = 0
        probs = trough_prior.dot(beta_probs)
        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(beta_probs[:n_thresholds_below_min])
        yin_probs[trough_index, i] = probs
    yin_period, frame_index = np.nonzero(yin_probs)
    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates
    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)
    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]
    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1)
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins
    return (observation_probs[np.newaxis], voiced_prob)
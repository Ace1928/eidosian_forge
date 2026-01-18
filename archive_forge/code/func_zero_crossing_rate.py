import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from .. import util
from .. import filters
from ..util.exceptions import ParameterError
from ..core.convert import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import power_to_db, _spectrogram
from ..core.constantq import cqt, hybrid_cqt, vqt
from ..core.pitch import estimate_tuning
from typing import Any, Optional, Union, Collection
from numpy.typing import DTypeLike
from .._typing import _FloatLike_co, _WindowSpec, _PadMode, _PadModeSTFT
def zero_crossing_rate(y: np.ndarray, *, frame_length: int=2048, hop_length: int=512, center: bool=True, **kwargs: Any) -> np.ndarray:
    """Compute the zero-crossing rate of an audio time series.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        Audio time series. Multi-channel is supported.
    frame_length : int > 0
        Length of the frame over which to compute zero crossing rates
    hop_length : int > 0
        Number of samples to advance for each frame
    center : bool
        If `True`, frames are centered by padding the edges of ``y``.
        This is similar to the padding in `librosa.stft`,
        but uses edge-value copies instead of zero-padding.
    **kwargs : additional keyword arguments to pass to `librosa.zero_crossings`
    threshold : float >= 0
        If specified, values where ``-threshold <= y <= threshold`` are
        clipped to 0.
    ref_magnitude : float > 0 or callable
        If numeric, the threshold is scaled relative to ``ref_magnitude``.
        If callable, the threshold is scaled relative to
        ``ref_magnitude(np.abs(y))``.
    zero_pos : boolean
        If ``True`` then the value 0 is interpreted as having positive sign.
        If ``False``, then 0, -1, and +1 all have distinct signs.
    axis : int
        Axis along which to compute zero-crossings.
        .. note:: By default, the ``pad`` parameter is set to `False`, which
        differs from the default specified by
        `librosa.zero_crossings`.

    Returns
    -------
    zcr : np.ndarray [shape=(..., 1, t)]
        ``zcr[..., 0, i]`` is the fraction of zero crossings in frame ``i``

    See Also
    --------
    librosa.zero_crossings : Compute zero-crossings in a time-series

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.zero_crossing_rate(y)
    array([[0.044, 0.074, ..., 0.488, 0.355]])
    """
    util.valid_audio(y, mono=False)
    if center:
        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode='edge')
    y_framed = util.frame(y, frame_length=frame_length, hop_length=hop_length)
    kwargs['axis'] = -2
    kwargs.setdefault('pad', False)
    crossings = zero_crossings(y_framed, **kwargs)
    zcrate: np.ndarray = np.mean(crossings, axis=-2, keepdims=True)
    return zcrate
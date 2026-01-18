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
def tonnetz(*, y: Optional[np.ndarray]=None, sr: float=22050, chroma: Optional[np.ndarray]=None, **kwargs: Any) -> np.ndarray:
    """Compute the tonal centroid features (tonnetz)

    This representation uses the method of [#]_ to project chroma features
    onto a 6-dimensional basis representing the perfect fifth, minor third,
    and major third each as two-dimensional coordinates.

    .. [#] Harte, C., Sandler, M., & Gasser, M. (2006). "Detecting Harmonic
           Change in Musical Audio." In Proceedings of the 1st ACM Workshop
           on Audio and Music Computing Multimedia (pp. 21-26).
           Santa Barbara, CA, USA: ACM Press. doi:10.1145/1178723.1178727.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        Audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    chroma : np.ndarray [shape=(n_chroma, t)] or None
        Normalized energy for each chroma bin at each frame.
        If `None`, a cqt chromagram is performed.
    **kwargs : Additional keyword arguments to `chroma_cqt`,
        if ``chroma`` is not pre-computed.
    C : np.ndarray [shape=(..., d, t)] [Optional]
        a pre-computed constant-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.
    tuning : float [scalar] or None.
        Deviation (in fractions of a CQT bin) from A440 tuning
    n_chroma : int > 0
        Number of chroma bins to produce
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`
    bins_per_octave : int > 0, optional
        Number of bins per octave in the CQT.
        Must be an integer multiple of ``n_chroma``.
        Default: 36 (3 bins per semitone)
        If `None`, it will match ``n_chroma``.
    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode

    Returns
    -------
    tonnetz : np.ndarray [shape(..., 6, t)]
        Tonal centroid features for each frame.

        Tonnetz dimensions:
            - 0: Fifth x-axis
            - 1: Fifth y-axis
            - 2: Minor x-axis
            - 3: Minor y-axis
            - 4: Major x-axis
            - 5: Major y-axis

    See Also
    --------
    chroma_cqt : Compute a chromagram from a constant-Q transform.
    chroma_stft : Compute a chromagram from an STFT spectrogram or waveform.

    Examples
    --------
    Compute tonnetz features from the harmonic component of a song

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=10, offset=10)
    >>> y = librosa.effects.harmonic(y)
    >>> tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    >>> tonnetz
    array([[ 0.007, -0.026, ...,  0.055,  0.056],
           [-0.01 , -0.009, ..., -0.012, -0.017],
           ...,
           [ 0.006, -0.021, ..., -0.012, -0.01 ],
           [-0.009,  0.031, ..., -0.05 , -0.037]])

    Compare the tonnetz features to `chroma_cqt`

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(tonnetz,
    ...                                 y_axis='tonnetz', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Tonal Centroids (Tonnetz)')
    >>> ax[0].label_outer()
    >>> img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),
    ...                                 y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Chroma')
    >>> fig.colorbar(img1, ax=[ax[0]])
    >>> fig.colorbar(img2, ax=[ax[1]])
    """
    if y is None and chroma is None:
        raise ParameterError('Either the audio samples or the chromagram must be passed as an argument.')
    if chroma is None:
        chroma = chroma_cqt(y=y, sr=sr, **kwargs)
    dim_map = np.linspace(0, 12, num=chroma.shape[-2], endpoint=False)
    scale = np.asarray([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])
    V = np.multiply.outer(scale, dim_map)
    V[::2] -= 0.5
    R = np.array([1, 1, 1, 1, 0.5, 0.5])
    phi = R[:, np.newaxis] * np.cos(np.pi * V)
    ton: np.ndarray = np.einsum('pc,...ci->...pi', phi, util.normalize(chroma, norm=1, axis=-2), optimize=True)
    return ton
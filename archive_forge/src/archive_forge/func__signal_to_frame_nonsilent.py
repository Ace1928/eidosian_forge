import numpy as np
import scipy.signal
from . import core
from . import decompose
from . import feature
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, Iterable, Optional, Tuple, Union, overload
from typing_extensions import Literal
from numpy.typing import ArrayLike
def _signal_to_frame_nonsilent(y: np.ndarray, frame_length: int=2048, hop_length: int=512, top_db: float=60, ref: Union[Callable, float]=np.max, aggregate: Callable=np.max) -> np.ndarray:
    """Frame-wise non-silent indicator for audio input.

    This is a helper function for `trim` and `split`.

    Parameters
    ----------
    y : np.ndarray
        Audio signal, mono or stereo

    frame_length : int > 0
        The number of samples per frame

    hop_length : int > 0
        The number of samples between frames

    top_db : number > 0
        The threshold (in decibels) below reference to consider as
        silence

    ref : callable or float
        The reference amplitude

    aggregate : callable [default: np.max]
        Function to aggregate dB measurements across channels (if y.ndim > 1)

        Note: for multiple leading axes, this is performed using ``np.apply_over_axes``.

    Returns
    -------
    non_silent : np.ndarray, shape=(m,), dtype=bool
        Indicator of non-silent frames
    """
    mse = feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    db = core.amplitude_to_db(mse[..., 0, :], ref=ref, top_db=None)
    if db.ndim > 1:
        db = np.apply_over_axes(aggregate, db, range(db.ndim - 1))
        db = np.squeeze(db, axis=tuple(range(db.ndim - 1)))
    return db > -top_db
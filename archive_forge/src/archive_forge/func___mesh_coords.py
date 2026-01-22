from __future__ import annotations
from itertools import product
import warnings
import numpy as np
import matplotlib.cm as mcm
import matplotlib.axes as mplaxes
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt
from . import core
from . import util
from .util.deprecation import rename_kw, Deprecated
from .util.exceptions import ParameterError
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Callable, Dict
from ._typing import _FloatLike_co
def __mesh_coords(ax_type, coords, n, **kwargs):
    """Compute axis coordinates"""
    if coords is not None:
        if len(coords) not in (n, n + 1):
            raise ParameterError(f'Coordinate shape mismatch: {len(coords)}!={n} or {n}+1')
        return coords
    coord_map: Dict[Optional[str], Callable[..., np.ndarray]] = {'linear': __coord_fft_hz, 'fft': __coord_fft_hz, 'fft_note': __coord_fft_hz, 'fft_svara': __coord_fft_hz, 'hz': __coord_fft_hz, 'log': __coord_fft_hz, 'mel': __coord_mel_hz, 'cqt': __coord_cqt_hz, 'cqt_hz': __coord_cqt_hz, 'cqt_note': __coord_cqt_hz, 'cqt_svara': __coord_cqt_hz, 'vqt_fjs': __coord_vqt_hz, 'vqt_hz': __coord_vqt_hz, 'vqt_note': __coord_vqt_hz, 'chroma': __coord_chroma, 'chroma_c': __coord_chroma, 'chroma_h': __coord_chroma, 'chroma_fjs': __coord_n, 'time': __coord_time, 'h': __coord_time, 'm': __coord_time, 's': __coord_time, 'ms': __coord_time, 'lag': __coord_time, 'lag_h': __coord_time, 'lag_m': __coord_time, 'lag_s': __coord_time, 'lag_ms': __coord_time, 'tonnetz': __coord_n, 'off': __coord_n, 'tempo': __coord_tempo, 'fourier_tempo': __coord_fourier_tempo, 'frames': __coord_n, None: __coord_n}
    if ax_type not in coord_map:
        raise ParameterError(f'Unknown axis type: {ax_type}')
    return coord_map[ax_type](n, **kwargs)
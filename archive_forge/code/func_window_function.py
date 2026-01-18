import warnings
from typing import Optional, Union
import numpy as np
def window_function(window_length: int, name: str='hann', periodic: bool=True, frame_length: Optional[int]=None, center: bool=True) -> np.ndarray:
    """
    Returns an array containing the specified window. This window is intended to be used with `stft`.

    The following window types are supported:

        - `"boxcar"`: a rectangular window
        - `"hamming"`: the Hamming window
        - `"hann"`: the Hann window
        - `"povey"`: the Povey window

    Args:
        window_length (`int`):
            The length of the window in samples.
        name (`str`, *optional*, defaults to `"hann"`):
            The name of the window function.
        periodic (`bool`, *optional*, defaults to `True`):
            Whether the window is periodic or symmetric.
        frame_length (`int`, *optional*):
            The length of the analysis frames in samples. Provide a value for `frame_length` if the window is smaller
            than the frame length, so that it will be zero-padded.
        center (`bool`, *optional*, defaults to `True`):
            Whether to center the window inside the FFT buffer. Only used when `frame_length` is provided.

    Returns:
        `np.ndarray` of shape `(window_length,)` or `(frame_length,)` containing the window.
    """
    length = window_length + 1 if periodic else window_length
    if name == 'boxcar':
        window = np.ones(length)
    elif name in ['hamming', 'hamming_window']:
        window = np.hamming(length)
    elif name in ['hann', 'hann_window']:
        window = np.hanning(length)
    elif name in ['povey']:
        window = np.power(np.hanning(length), 0.85)
    else:
        raise ValueError(f"Unknown window function '{name}'")
    if periodic:
        window = window[:-1]
    if frame_length is None:
        return window
    if window_length > frame_length:
        raise ValueError(f'Length of the window ({window_length}) may not be larger than frame_length ({frame_length})')
    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset:offset + window_length] = window
    return padded_window
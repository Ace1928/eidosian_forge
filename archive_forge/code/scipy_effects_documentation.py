from scipy.signal import butter, sosfilt
from .utils import (register_pydub_effect,stereo_to_ms,ms_to_stereo)

    Args:
        focus_freq - middle frequency or known frequency of band (in Hz)
        bandwidth - range of the equalizer band
        channel_mode - Select Channels to be affected by the filter.
            L+R - Standard Stereo Filter
            L - Only Left Channel is Filtered
            R - Only Right Channel is Filtered
            M+S - Blumlien Stereo Filter(Mid-Side)
            M - Only Mid Channel is Filtered
            S - Only Side Channel is Filtered
            Mono Audio Segments are completely filtered.
        filter_mode - Mode of Equalization(Peak/Notch(Bell Curve),High Shelf, Low Shelf)
        order - Rolloff factor(1 - 6dB/Octave 2 - 12dB/Octave)
    
    Returns:
        Equalized/Filtered AudioSegment
    
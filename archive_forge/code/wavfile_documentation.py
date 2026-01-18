import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum

    Write a NumPy array as a WAV file.

    Parameters
    ----------
    filename : string or open file handle
        Output wav file.
    rate : int
        The sample rate (in samples/sec).
    data : ndarray
        A 1-D or 2-D NumPy array of either integer or float data-type.

    Notes
    -----
    * Writes a simple uncompressed WAV file.
    * To write multiple-channels, use a 2-D array of shape
      (Nsamples, Nchannels).
    * The bits-per-sample and PCM/float will be determined by the data-type.

    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    Note that 8-bit PCM is unsigned.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    Examples
    --------
    Create a 100Hz sine wave, sampled at 44100Hz.
    Write to 16-bit PCM, Mono.

    >>> from scipy.io.wavfile import write
    >>> import numpy as np
    >>> samplerate = 44100; fs = 100
    >>> t = np.linspace(0., 1., samplerate)
    >>> amplitude = np.iinfo(np.int16).max
    >>> data = amplitude * np.sin(2. * np.pi * fs * t)
    >>> write("example.wav", samplerate, data.astype(np.int16))

    
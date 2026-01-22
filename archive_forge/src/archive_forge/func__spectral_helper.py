import warnings
import cupy
from cupy_backends.cuda.api import runtime
import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal.windows._windows import get_window
def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd', boundary=None, padded=False):
    """
    Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab

    """
    if mode not in ['psd', 'stft']:
        raise ValueError(f"Unknown value for mode {mode}, must be one of: {{'psd', 'stft'}}")
    boundary_funcs = {'even': even_ext, 'odd': odd_ext, 'constant': const_ext, 'zeros': zero_ext, None: None}
    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{0}', must be one of: {1}".format(boundary, list(boundary_funcs.keys())))
    same_data = y is x
    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")
    axis = int(axis)
    x = cupy.asarray(x)
    if not same_data:
        y = cupy.asarray(y)
        outdtype = cupy.result_type(x, y, cupy.complex64)
    else:
        outdtype = cupy.result_type(x, cupy.complex64)
    if not same_data:
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = cupy.broadcast(cupy.empty(xouter), cupy.empty(youter)).shape
        except ValueError:
            raise ValueError('x and y cannot be broadcast together.')
    if same_data:
        if x.size == 0:
            return (cupy.empty(x.shape), cupy.empty(x.shape), cupy.empty(x.shape))
    elif x.size == 0 or y.size == 0:
        outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
        emptyout = cupy.rollaxis(cupy.empty(outshape), -1, axis)
        return (emptyout, emptyout, emptyout)
    if x.ndim > 1:
        if axis != -1:
            x = cupy.rollaxis(x, axis, len(x.shape))
            if not same_data and y.ndim > 1:
                y = cupy.rollaxis(y, axis, len(y.shape))
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = cupy.concatenate((x, cupy.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = cupy.concatenate((y, cupy.zeros(pad_shape)), -1)
    if nperseg is not None:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])
    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap
    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)
    if padded:
        nadd = -(x.shape[-1] - nperseg) % nstep % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = cupy.concatenate((x, cupy.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = cupy.concatenate((y, cupy.zeros(zeros_shape)), axis=-1)
    if not detrend:

        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):

        def detrend_func(d):
            return filtering.detrend(d, type=detrend, axis=-1)
    elif axis != -1:

        def detrend_func(d):
            d = cupy.rollaxis(d, -1, axis)
            d = detrend(d)
            return cupy.rollaxis(d, axis, len(d.shape))
    else:
        detrend_func = detrend
    if cupy.result_type(win, cupy.complex64) != outdtype:
        win = win.astype(outdtype)
    if scaling == 'density':
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)
    if mode == 'stft':
        scale = cupy.sqrt(scale)
    if return_onesided:
        if cupy.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if cupy.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to return_onesided=False')
    else:
        sides = 'twosided'
    if sides == 'twosided':
        freqs = cupy.fft.fftfreq(nfft, 1 / fs)
    elif sides == 'onesided':
        freqs = cupy.fft.rfftfreq(nfft, 1 / fs)
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)
    if not same_data:
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft, sides)
        result = cupy.conj(result) * result_y
    elif mode == 'psd':
        result = cupy.conj(result) * result
    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            result[..., 1:-1] *= 2
    time = cupy.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap) / float(fs)
    if boundary is not None:
        time -= nperseg / 2 / fs
    result = result.astype(outdtype)
    if same_data and mode != 'stft':
        result = result.real
    if axis < 0:
        axis -= 1
    result = cupy.rollaxis(result, -1, axis)
    return (freqs, time, result)
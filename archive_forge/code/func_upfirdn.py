from math import ceil
import cupy
def upfirdn(h, x, up=1, down=1, axis=-1, mode=None, cval=0):
    """
    Upsample, FIR filter, and downsample.

    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.
    mode : str, optional
        This parameter is not implemented.
    cval : float, optional
        This parameter is not implemented.

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    See Also
    --------
    scipy.signal.upfirdn

    References
    ----------
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    """
    if mode is not None or cval != 0:
        raise NotImplementedError(f'mode = {mode!r} and cval ={cval!r} not implemented.')
    ufd = _UpFIRDn(h, x.dtype, int(up), int(down))
    return ufd.apply_filter(x, axis)
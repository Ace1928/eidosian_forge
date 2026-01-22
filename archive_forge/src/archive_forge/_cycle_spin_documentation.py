from itertools import product
import numpy as np
from .._shared import utils
from .._shared.utils import warn
Cycle spinning (repeatedly apply func to shifted versions of x).

    Parameters
    ----------
    x : array-like
        Data for input to ``func``.
    func : function
        A function to apply to circularly shifted versions of ``x``.  Should
        take ``x`` as its first argument. Any additional arguments can be
        supplied via ``func_kw``.
    max_shifts : int or tuple
        If an integer, shifts in ``range(0, max_shifts+1)`` will be used along
        each axis of ``x``. If a tuple, ``range(0, max_shifts[i]+1)`` will be
        along axis i.
    shift_steps : int or tuple, optional
        The step size for the shifts applied along axis, i, are::
        ``range((0, max_shifts[i]+1, shift_steps[i]))``. If an integer is
        provided, the same step size is used for all axes.
    num_workers : int or None, optional
        The number of parallel threads to use during cycle spinning. If set to
        ``None``, the full set of available cores are used.
    func_kw : dict, optional
        Additional keyword arguments to supply to ``func``.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    avg_y : np.ndarray
        The output of ``func(x, **func_kw)`` averaged over all combinations of
        the specified axis shifts.

    Notes
    -----
    Cycle spinning was proposed as a way to approach shift-invariance via
    performing several circular shifts of a shift-variant transform [1]_.

    For a n-level discrete wavelet transforms, one may wish to perform all
    shifts up to ``max_shifts = 2**n - 1``. In practice, much of the benefit
    can often be realized with only a small number of shifts per axis.

    For transforms such as the blockwise discrete cosine transform, one may
    wish to evaluate shifts up to the block size used by the transform.

    References
    ----------
    .. [1] R.R. Coifman and D.L. Donoho.  "Translation-Invariant De-Noising".
           Wavelets and Statistics, Lecture Notes in Statistics, vol.103.
           Springer, New York, 1995, pp.125-150.
           :DOI:`10.1007/978-1-4612-2544-7_9`

    Examples
    --------
    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> from skimage.restoration import denoise_tv_chambolle, cycle_spin
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> img = img + sigma * np.random.standard_normal(img.shape)
    >>> denoised = cycle_spin(img, func=denoise_tv_chambolle,
    ...                       max_shifts=3)

    
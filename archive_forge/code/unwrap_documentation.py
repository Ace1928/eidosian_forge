import numpy as np
from .._shared.utils import warn
from ._unwrap_1d import unwrap_1d
from ._unwrap_2d import unwrap_2d
from ._unwrap_3d import unwrap_3d
Recover the original from a wrapped phase image.

    From an image wrapped to lie in the interval [-pi, pi), recover the
    original, unwrapped image.

    Parameters
    ----------
    image : (M[, N[, P]]) ndarray or masked array of floats
        The values should be in the range [-pi, pi). If a masked array is
        provided, the masked entries will not be changed, and their values
        will not be used to guide the unwrapping of neighboring, unmasked
        values. Masked 1D arrays are not allowed, and will raise a
        `ValueError`.
    wrap_around : bool or sequence of bool, optional
        When an element of the sequence is  `True`, the unwrapping process
        will regard the edges along the corresponding axis of the image to be
        connected and use this connectivity to guide the phase unwrapping
        process. If only a single boolean is given, it will apply to all axes.
        Wrap around is not supported for 1D arrays.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

        Unwrapping relies on a random initialization. This sets the
        PRNG to use to achieve deterministic behavior.

    Returns
    -------
    image_unwrapped : array_like, double
        Unwrapped image of the same shape as the input. If the input `image`
        was a masked array, the mask will be preserved.

    Raises
    ------
    ValueError
        If called with a masked 1D array or called with a 1D array and
        ``wrap_around=True``.

    Examples
    --------
    >>> c0, c1 = np.ogrid[-1:1:128j, -1:1:128j]
    >>> image = 12 * np.pi * np.exp(-(c0**2 + c1**2))
    >>> image_wrapped = np.angle(np.exp(1j * image))
    >>> image_unwrapped = unwrap_phase(image_wrapped)
    >>> np.std(image_unwrapped - image) < 1e-6   # A constant offset is normal
    True

    References
    ----------
    .. [1] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
           and Munther A. Gdeisat, "Fast two-dimensional phase-unwrapping
           algorithm based on sorting by reliability following a noncontinuous
           path", Journal Applied Optics, Vol. 41, No. 35 (2002) 7437,
    .. [2] Abdul-Rahman, H., Gdeisat, M., Burton, D., & Lalor, M., "Fast
           three-dimensional phase-unwrapping algorithm based on sorting by
           reliability following a non-continuous path. In W. Osten,
           C. Gorecki, & E. L. Novak (Eds.), Optical Metrology (2005) 32--40,
           International Society for Optics and Photonics.
    
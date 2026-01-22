from itertools import product
import numpy as np
def _coefficients_no_filter(f, degree, use_broadcasting):
    """Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\\pi` periodic
    function, where :math:`d` is the highest desired frequency in the Fourier spectrum.

    This function computes the coefficients blindly without any filtering applied, and
    is thus used as a helper function for the true ``coefficients`` function.

    Args:
        f (callable): function that takes a 1D array of scalar inputs
        degree (int or tuple[int]): max frequency of Fourier coeffs to be computed. For degree
            :math:`d`, the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d`
            will be computed.
        use_broadcasting (bool): Whether or not to broadcast the parameters to execute
            multiple function calls at once. Broadcasting is performed along the last axis
            of the grid of evaluation points.

    Returns:
        array[complex]: The Fourier coefficients of the function f up to the specified degree.
    """
    degree = np.array(degree)
    k = 2 * degree + 1
    n_ranges = [np.arange(-d, d + 1) for d in degree]
    nvecs = product(*(n_ranges[:-1] if use_broadcasting else n_ranges))
    f_discrete = np.zeros(shape=tuple(k))
    spacing = 2 * np.pi / k
    for nvec in nvecs:
        if use_broadcasting:
            nvec = (*nvec, n_ranges[-1])
            sampling_point = [s * n for s, n in zip(spacing, nvec)]
        else:
            sampling_point = spacing * np.array(nvec)
        f_discrete[nvec] = f(sampling_point)
    coeffs = np.fft.fftn(f_discrete) / f_discrete.size
    return coeffs
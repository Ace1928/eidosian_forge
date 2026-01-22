import math
from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window
import cupy
import numpy
Convert a linear-phase FIR filter to minimum phase

    Parameters
    ----------
    h : array
        Linear-phase FIR filter coefficients.
    method : {'hilbert', 'homomorphic'}
        The method to use:

            'homomorphic' (default)
                This method [4]_ [5]_ works best with filters with an
                odd number of taps, and the resulting minimum phase filter
                will have a magnitude response that approximates the square
                root of the original filter's magnitude response.

            'hilbert'
                This method [1]_ is designed to be used with equiripple
                filters (e.g., from `remez`) with unity or zero gain
                regions.

    n_fft : int
        The number of points to use for the FFT. Should be at least a
        few times larger than the signal length (see Notes).

    Returns
    -------
    h_minimum : array
        The minimum-phase version of the filter, with length
        ``(length(h) + 1) // 2``.

    See Also
    --------
    scipy.signal.minimum_phase

    Notes
    -----
    Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection
    of an FFT length to estimate the complex cepstrum of the filter.

    In the case of the Hilbert method, the deviation from the ideal
    spectrum ``epsilon`` is related to the number of stopband zeros
    ``n_stop`` and FFT length ``n_fft`` as::

        epsilon = 2. * n_stop / n_fft

    For example, with 100 stopband zeros and a FFT length of 2048,
    ``epsilon = 0.0976``. If we conservatively assume that the number of
    stopband zeros is one less than the filter length, we can take the FFT
    length to be the next power of 2 that satisfies ``epsilon=0.01`` as::

        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))

    This gives reasonable results for both the Hilbert and homomorphic
    methods, and gives the value used when ``n_fft=None``.

    Alternative implementations exist for creating minimum-phase filters,
    including zero inversion [2]_ and spectral factorization [3]_ [4]_ [5]_.
    For more information, see:

        http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters

    References
    ----------
    .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and
           complex minimum phase digital FIR filters," Acoustics, Speech,
           and Signal Processing, 1999. Proceedings., 1999 IEEE International
           Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.
           DOI:10.1109/ICASSP.1999.756179
    .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR
           filters by direct factorization," Signal Processing,
           vol. 10, no. 4, pp. 369-383, Jun. 1986.
    .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in
           Handbook for Digital Signal Processing, chapter 4,
           New York: Wiley-Interscience, 1993.
    .. [4] J. S. Lim, Advanced Topics in Signal Processing.
           Englewood Cliffs, N.J.: Prentice Hall, 1988.
    .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,
           "Discrete-Time Signal Processing," 2nd edition.
           Upper Saddle River, N.J.: Prentice Hall, 1999.

    
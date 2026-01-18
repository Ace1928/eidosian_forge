import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.core.overrides import array_function_dispatch
from numpy.lib.type_check import isreal

    Compute the inverse hyperbolic tangent of `x`.

    Return the "principal value" (for a description of this, see
    `numpy.arctanh`) of ``arctanh(x)``. For real `x` such that
    ``abs(x) < 1``, this is a real number.  If `abs(x) > 1`, or if `x` is
    complex, the result is complex. Finally, `x = 1` returns``inf`` and
    ``x=-1`` returns ``-inf``.

    Parameters
    ----------
    x : array_like
       The value(s) whose arctanh is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The inverse hyperbolic tangent(s) of the `x` value(s). If `x` was
       a scalar so is `out`, otherwise an array is returned.


    See Also
    --------
    numpy.arctanh

    Notes
    -----
    For an arctanh() that returns ``NAN`` when real `x` is not in the
    interval ``(-1,1)``, use `numpy.arctanh` (this latter, however, does
    return +/-inf for ``x = +/-1``).

    Examples
    --------
    >>> np.set_printoptions(precision=4)

    >>> from numpy.testing import suppress_warnings
    >>> with suppress_warnings() as sup:
    ...     sup.filter(RuntimeWarning)
    ...     np.emath.arctanh(np.eye(2))
    array([[inf,  0.],
           [ 0., inf]])
    >>> np.emath.arctanh([1j])
    array([0.+0.7854j])

    
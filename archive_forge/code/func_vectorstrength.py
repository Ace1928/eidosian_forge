import warnings
import cupy
from cupyx.scipy.signal.windows._windows import get_window
from cupyx.scipy.signal._spectral_impl import (
def vectorstrength(events, period):
    """
    Determine the vector strength of the events corresponding to the given
    period.

    The vector strength is a measure of phase synchrony, how well the
    timing of the events is synchronized to a single period of a periodic
    signal.

    If multiple periods are used, calculate the vector strength of each.
    This is called the "resonating vector strength".

    Parameters
    ----------
    events : 1D array_like
        An array of time points containing the timing of the events.
    period : float or array_like
        The period of the signal that the events should synchronize to.
        The period is in the same units as `events`.  It can also be an array
        of periods, in which case the outputs are arrays of the same length.

    Returns
    -------
    strength : float or 1D array
        The strength of the synchronization.  1.0 is perfect synchronization
        and 0.0 is no synchronization.  If `period` is an array, this is also
        an array with each element containing the vector strength at the
        corresponding period.
    phase : float or array
        The phase that the events are most strongly synchronized to in radians.
        If `period` is an array, this is also an array with each element
        containing the phase for the corresponding period.

    Notes
    -----
    See [1]_, [2]_ and [3]_ for more information.

    References
    ----------
    .. [1] van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating
           vector strength: Auditory system, electric fish, and noise.
           Chaos 21, 047508 (2011).
    .. [2] van Hemmen, JL. Vector strength after Goldberg, Brown, and
           von Mises: biological and mathematical perspectives.  Biol Cybern.
           2013 Aug;107(4):385-96.
    .. [3] van Hemmen, JL and Vollmayr, AN.  Resonating vector strength:
           what happens when we vary the "probing" frequency while keeping
           the spike times fixed.  Biol Cybern. 2013 Aug;107(4):491-94.
    """
    events = cupy.asarray(events)
    period = cupy.asarray(period)
    if events.ndim > 1:
        raise ValueError('events cannot have dimensions more than 1')
    if period.ndim > 1:
        raise ValueError('period cannot have dimensions more than 1')
    scalarperiod = not period.ndim
    events = cupy.atleast_2d(events)
    period = cupy.atleast_2d(period)
    if (period <= 0).any():
        raise ValueError('periods must be positive')
    vectors = cupy.exp(cupy.dot(2j * cupy.pi / period.T, events))
    vectormean = cupy.mean(vectors, axis=1)
    strength = cupy.abs(vectormean)
    phase = cupy.angle(vectormean)
    if scalarperiod:
        strength = strength[0]
        phase = phase[0]
    return (strength, phase)
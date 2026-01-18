import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def zpk2sos(z, p, k, pairing=None, *, analog=False):
    """Return second-order sections from zeros, poles, and gain of a system

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    pairing : {None, 'nearest', 'keep_odd', 'minimal'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        If analog is False and pairing is None, pairing is set to 'nearest';
        if analog is True, pairing must be 'minimal', and is set to that if
        it is None.
    analog : bool, optional
        If True, system is analog, otherwise discrete.

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    sosfilt
    scipy.signal.zpk2sos

    """
    if pairing is None:
        pairing = 'minimal' if analog else 'nearest'
    valid_pairings = ['nearest', 'keep_odd', 'minimal']
    if pairing not in valid_pairings:
        raise ValueError('pairing must be one of %s, not %s' % (valid_pairings, pairing))
    if analog and pairing != 'minimal':
        raise ValueError('for analog zpk2sos conversion, pairing must be "minimal"')
    if len(z) == len(p) == 0:
        if not analog:
            return cupy.array([[k, 0.0, 0.0, 1.0, 0.0, 0.0]])
        else:
            return cupy.array([[0.0, 0.0, k, 0.0, 0.0, 1.0]])
    if pairing != 'minimal':
        p = cupy.concatenate((p, cupy.zeros(max(len(z) - len(p), 0))))
        z = cupy.concatenate((z, cupy.zeros(max(len(p) - len(z), 0))))
        n_sections = (max(len(p), len(z)) + 1) // 2
        if len(p) % 2 == 1 and pairing == 'nearest':
            p = cupy.concatenate((p, cupy.zeros(1)))
            z = cupy.concatenate((z, cupy.zeros(1)))
        assert len(p) == len(z)
    else:
        if len(p) < len(z):
            raise ValueError('for analog zpk2sos conversion, must have len(p)>=len(z)')
        n_sections = (len(p) + 1) // 2
    z = cupy.concatenate(_cplxreal(z))
    p = cupy.concatenate(_cplxreal(p))
    if not cupy.isreal(k):
        raise ValueError('k must be real')
    k = k.real
    if not analog:

        def idx_worst(p):
            return cupy.argmin(cupy.abs(1 - cupy.abs(p)))
    else:

        def idx_worst(p):
            return cupy.argmin(cupy.abs(cupy.real(p)))
    sos = cupy.zeros((n_sections, 6))
    for si in range(n_sections - 1, -1, -1):
        p1_idx = idx_worst(p)
        p1 = p[p1_idx]
        p = cupy.delete(p, p1_idx)
        if cupy.isreal(p1) and cupy.isreal(p).sum() == 0:
            if pairing != 'minimal':
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = cupy.delete(z, z1_idx)
                sos[si] = _single_zpksos(cupy.r_[z1, 0], cupy.r_[p1, 0], 1)
            elif len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'real')
                z1 = z[z1_idx]
                z = cupy.delete(z, z1_idx)
                sos[si] = _single_zpksos([z1], [p1], 1)
            else:
                sos[si] = _single_zpksos([], [p1], 1)
        elif len(p) + 1 == len(z) and (not cupy.isreal(p1)) and (cupy.isreal(p).sum() == 1) and (cupy.isreal(z).sum() == 1):
            z1_idx = _nearest_real_complex_idx(z, p1, 'complex')
            z1 = z[z1_idx]
            z = cupy.delete(z, z1_idx)
            sos[si] = _single_zpksos(cupy.r_[z1, z1.conj()], cupy.r_[p1, p1.conj()], 1)
        else:
            if cupy.isreal(p1):
                prealidx = cupy.flatnonzero(cupy.isreal(p))
                p2_idx = prealidx[idx_worst(p[prealidx])]
                p2 = p[p2_idx]
                p = cupy.delete(p, p2_idx)
            else:
                p2 = p1.conj()
            if len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, 'any')
                z1 = z[z1_idx]
                z = cupy.delete(z, z1_idx)
                if not cupy.isreal(z1):
                    sos[si] = _single_zpksos(cupy.r_[z1, z1.conj()], cupy.r_[p1, p2], 1)
                elif len(z) > 0:
                    z2_idx = _nearest_real_complex_idx(z, p1, 'real')
                    z2 = z[z2_idx]
                    assert cupy.isreal(z2)
                    z = cupy.delete(z, z2_idx)
                    sos[si] = _single_zpksos(cupy.r_[z1, z2], [p1, p2], 1)
                else:
                    sos[si] = _single_zpksos([z1], [p1, p2], 1)
            else:
                sos[si] = _single_zpksos([], [p1, p2], 1)
    assert len(p) == len(z) == 0
    del p, z
    sos[0][:3] *= k
    return sos
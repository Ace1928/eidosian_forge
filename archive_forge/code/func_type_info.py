from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def type_info(np_type):
    """Return dict with min, max, nexp, nmant, width for numpy type `np_type`

    Type can be integer in which case nexp and nmant are None.

    Parameters
    ----------
    np_type : numpy type specifier
        Any specifier for a numpy dtype

    Returns
    -------
    info : dict
        with fields ``min`` (minimum value), ``max`` (maximum value), ``nexp``
        (exponent width), ``nmant`` (significand precision not including
        implicit first digit), ``minexp`` (minimum exponent), ``maxexp``
        (maximum exponent), ``width`` (width in bytes). (``nexp``, ``nmant``,
        ``minexp``, ``maxexp``) are None for integer types. Both ``min`` and
        ``max`` are of type `np_type`.

    Raises
    ------
    FloatingError
        for floating point types we don't recognize

    Notes
    -----
    You might be thinking that ``np.finfo`` does this job, and it does, except
    for PPC long doubles (https://github.com/numpy/numpy/issues/2669) and
    float96 on Windows compiled with Mingw. This routine protects against such
    errors in ``np.finfo`` by only accepting values that we know are likely to
    be correct.
    """
    dt = np.dtype(np_type)
    np_type = dt.type
    width = dt.itemsize
    try:
        info = np.iinfo(dt)
    except ValueError:
        pass
    else:
        return dict(min=np_type(info.min), max=np_type(info.max), minexp=None, maxexp=None, nmant=None, nexp=None, width=width)
    info = np.finfo(dt)
    nmant, nexp = (info.nmant, info.nexp)
    ret = dict(min=np_type(info.min), max=np_type(info.max), nmant=nmant, nexp=nexp, minexp=info.minexp, maxexp=info.maxexp, width=width)
    if np_type in (np.float16, np.float32, np.float64, np.complex64, np.complex128):
        return ret
    info_64 = np.finfo(np.float64)
    if dt.kind == 'c':
        assert np_type is np.clongdouble
        vals = (nmant, nexp, width / 2)
    else:
        assert np_type is np.longdouble
        vals = (nmant, nexp, width)
    if vals in ((112, 15, 16), (info_64.nmant, info_64.nexp, 8), (63, 15, 12), (63, 15, 16)):
        return ret
    ret = type_info(np.float64)
    if vals in ((52, 15, 12), (52, 15, 16)):
        ret.update(dict(width=width))
        return ret
    if vals == (105, 11, 16):
        ret.update(dict(nmant=nmant, nexp=nexp, width=width))
        return ret
    if np_type not in (np.longdouble, np.clongdouble) or width not in (16, 32):
        raise FloatingError(f'We had not expected type {np_type}')
    if vals == (1, 1, 16) and on_powerpc() and _check_maxexp(np.longdouble, 1024):
        ret.update(dict(nmant=106, width=width))
    elif _check_nmant(np.longdouble, 52) and _check_maxexp(np.longdouble, 11):
        pass
    elif _check_nmant(np.longdouble, 112) and _check_maxexp(np.longdouble, 16384):
        two = np.longdouble(2)
        max_val = (two ** 113 - 1) / two ** 112 * two ** 16383
        if np_type is np.clongdouble:
            max_val += 0j
        ret = dict(min=-max_val, max=max_val, nmant=112, nexp=15, minexp=-16382, maxexp=16384, width=width)
    else:
        raise FloatingError(f'We had not expected long double type {np_type} with info {info}')
    return ret
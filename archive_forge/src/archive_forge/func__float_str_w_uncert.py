from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def _float_str_w_uncert(x, xe, precision=2):
    """Prints uncertain number with parenthesis

    Parameters
    ----------
    x : nominal value
    xe : uncertainty
    precision : number of significant digits in uncertainty

    Examples
    --------
    >>> _float_str_w_uncert(-9.99752e5, 349, 3)
    '-999752(349)'
    >>> _float_str_w_uncert(-9.99752e15, 349e10, 2)
    '-9.9975(35)e15'
    >>> _float_str_w_uncert(3.1416, 0.029, 1)
    '3.14(3)'
    >>> _float_str_w_uncert(3.1416e9, 2.9e6, 1)
    '3.142(3)e9'

    Returns
    -------
    shortest string representation of "x +- xe" either as
    ``x.xx(ee)e+xx`` or ``xxx.xx(ee)``

    Notes
    -----
    The code in this function is from a question on StackOverflow:
        http://stackoverflow.com/questions/6671053
        written by:
            Lemming, http://stackoverflow.com/users/841562/lemming
        the code is licensed under 'CC-WIKI'.
        (see: http://blog.stackoverflow.com/2009/06/attribution-required/)

    """
    x_exp = int(floor(log10(abs(x))))
    xe_exp = int(floor(log10(abs(xe))))
    un_exp = xe_exp - precision + 1
    un_int = round(xe * 10 ** (-un_exp))
    no_exp = un_exp
    no_int = round(x * 10 ** (-no_exp))
    fieldw = x_exp - no_exp
    fmt = '%%.%df' % fieldw
    result1 = (fmt + '(%.0f)e%d') % (no_int * 10 ** (-fieldw), un_int, x_exp)
    fieldw = max(0, -no_exp)
    fmt = '%%.%df' % fieldw
    result2 = (fmt + '(%.0f)') % (no_int * 10 ** no_exp, un_int * 10 ** max(0, un_exp))
    if len(result2) <= len(result1):
        return result2
    else:
        return result1
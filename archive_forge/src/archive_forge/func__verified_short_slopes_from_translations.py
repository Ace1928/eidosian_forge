from ..sage_helper import _within_sage
import math
def _verified_short_slopes_from_translations(translations, length=6):
    m_tran, l_tran = translations
    if not m_tran > 0:
        raise Exception('Expected positive meridian translation')
    RIF = m_tran.parent()
    length = RIF(length)
    result = []
    max_abs_l = _max_int_in_interval(length / abs(l_tran.imag()))
    for l in range(max_abs_l + 1):
        total_l_tran = l * l_tran
        max_real_range_sqr = (length ** 2 - total_l_tran.imag() ** 2).upper()
        if max_real_range_sqr >= 0:
            max_real_range = RIF(max_real_range_sqr).sqrt()
            if l == 0:
                min_m = 1
            else:
                min_m = _min_int_in_interval((-total_l_tran.real() - max_real_range) / m_tran)
            max_m = _max_int_in_interval((-total_l_tran.real() + max_real_range) / m_tran)
            for m in range(min_m, max_m + 1):
                if gcd(m, l) == 1:
                    result.append((m, l))
    return result
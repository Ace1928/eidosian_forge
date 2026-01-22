from ..sage_helper import _within_sage
import math
def _unverified_short_slopes_from_translations(translations, length=6):
    m_tran, l_tran = translations
    if isinstance(m_tran, complex):
        raise Exception('Expected real meridian translation')
    if not isinstance(m_tran, float):
        if m_tran.imag() != 0.0:
            raise Exception('Expected real meridian translation')
    if not m_tran > 0:
        raise Exception('Expected positive meridian translation')
    length = length * 1.001
    result = []
    max_abs_l = _floor(length / abs(_imag(l_tran)))
    for l in range(max_abs_l + 1):
        total_l_tran = l * l_tran
        max_real_range_sqr = length ** 2 - _imag(total_l_tran) ** 2
        if max_real_range_sqr >= 0:
            max_real_range = sqrt(max_real_range_sqr)
            if l == 0:
                min_m = 1
            else:
                min_m = _ceil((-_real(total_l_tran) - max_real_range) / m_tran)
            max_m = _floor((-_real(total_l_tran) + max_real_range) / m_tran)
            for m in range(min_m, max_m + 1):
                if gcd(m, l) in [-1, +1]:
                    result.append((m, l))
    return result
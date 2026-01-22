import numpy as np
def _get_conversion_multiplier(big_unit_code, small_unit_code):
    """
    Return an integer multiplier allowing to convert from *big_unit_code*
    to *small_unit_code*.
    None is returned if the conversion is not possible through a
    simple integer multiplication.
    """
    if big_unit_code == 14:
        return 1
    c = big_unit_code
    factor = 1
    while c < small_unit_code:
        try:
            c, mult = _factors[c]
        except KeyError:
            return None
        factor *= mult
    if c == small_unit_code:
        return factor
    else:
        return None
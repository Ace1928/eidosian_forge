def _encode_flips(flips):
    """
    Given flips, encode as bit field.
    """
    chars = ''
    padded_flips = flips + 5 * [0]
    for pos in range(len(padded_flips) // 6):
        val = 0
        for i in range(6):
            val |= _boolean_to_int(padded_flips[6 * pos + i]) << i
        chars += _unsigned_int_to_char(val)
    return chars
def _int_to_chars(value, num_chars):
    """
    Encode the given integer using base64-like encoding with num_chars
    characters.
    """
    abs_value = abs(value)
    chars = ''
    for pos in range(num_chars):
        if pos == num_chars - 1:
            if value < 0:
                abs_value |= 1 << 5
        chars = _unsigned_int_to_char(abs_value & 63) + chars
        abs_value >>= 6
    return chars
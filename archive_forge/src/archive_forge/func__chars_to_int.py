def _chars_to_int(chars):
    """
    Take a string of ASCII characters and convert it to integer using the
    base64-like scheme described above.
    """
    value = 0
    for pos, char in enumerate(chars):
        i = _char_to_unsigned_int(char)
        if pos == 0:
            if i & 1 << 5:
                sign = -1
            else:
                sign = +1
            i = i & 31
        value = (value << 6) + i
    return sign * value
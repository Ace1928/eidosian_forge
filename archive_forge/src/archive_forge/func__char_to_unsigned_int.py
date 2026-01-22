def _char_to_unsigned_int(char):
    """
    Convert an ASCII character to an integer 0-63
    """
    i = ord(char)
    if 97 <= i <= 122:
        return i - 97
    if 65 <= i <= 90:
        return i - 39
    if 48 <= i <= 57:
        return i + 4
    if i == 43:
        return 62
    return 63
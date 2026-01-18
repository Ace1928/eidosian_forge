def unctrl(c):
    bits = _ctoi(c)
    if bits == 127:
        rep = '^?'
    elif isprint(bits & 127):
        rep = chr(bits & 127)
    else:
        rep = '^' + chr((bits & 127 | 32) + 32)
    if bits & 128:
        return '!' + rep
    return rep
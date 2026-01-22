def ascii(c):
    if type(c) == type(''):
        return chr(_ctoi(c) & 127)
    else:
        return _ctoi(c) & 127
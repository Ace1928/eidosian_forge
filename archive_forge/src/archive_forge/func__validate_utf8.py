import six
def _validate_utf8(utfbytes):
    state = _UTF8_ACCEPT
    codep = 0
    for i in utfbytes:
        if six.PY2:
            i = ord(i)
        state, codep = _decode(state, codep, i)
        if state == _UTF8_REJECT:
            return False
    return True
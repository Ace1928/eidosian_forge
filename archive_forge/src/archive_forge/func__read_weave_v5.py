def _read_weave_v5(f, w):
    """Private helper routine to read a weave format 5 file into memory.

    This is only to be used by read_weave and WeaveFile.__init__.
    """
    from .weave import WeaveFormatError
    try:
        lines = iter(f.readlines())
    finally:
        f.close()
    try:
        l = next(lines)
    except StopIteration:
        raise WeaveFormatError('invalid weave file: no header')
    if l != FORMAT_1:
        raise WeaveFormatError('invalid weave file header: %r' % l)
    ver = 0
    while True:
        l = next(lines)
        if l[0:1] == b'i':
            if len(l) > 2:
                w._parents.append(list(map(int, l[2:].split(b' '))))
            else:
                w._parents.append([])
            l = next(lines)[:-1]
            w._sha1s.append(l[2:])
            l = next(lines)
            name = l[2:-1]
            w._names.append(name)
            w._name_map[name] = ver
            l = next(lines)
            ver += 1
        elif l == b'w\n':
            break
        else:
            raise WeaveFormatError('unexpected line %r' % l)
    while True:
        l = next(lines)
        if l == b'W\n':
            break
        elif b'. ' == l[0:2]:
            w._weave.append(l[2:])
        elif b', ' == l[0:2]:
            w._weave.append(l[2:-1])
        elif l == b'}\n':
            w._weave.append((b'}', None))
        else:
            w._weave.append((l[0:1], int(l[2:].decode('ascii'))))
    return w
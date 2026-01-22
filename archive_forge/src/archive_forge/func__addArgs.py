from fontTools.cffLib import maxStackLimit
def _addArgs(a, b):
    if isinstance(b, list):
        if isinstance(a, list):
            if len(a) != len(b) or a[-1] != b[-1]:
                raise ValueError()
            return [_addArgs(va, vb) for va, vb in zip(a[:-1], b[:-1])] + [a[-1]]
        else:
            a, b = (b, a)
    if isinstance(a, list):
        assert a[-1] == 1
        return [_addArgs(a[0], b)] + a[1:]
    return a + b
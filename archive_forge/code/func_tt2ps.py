def tt2ps(fn, b, i):
    """family name + bold & italic to ps font name"""
    K = (fn.lower(), b, i)
    if K in _tt2ps_map:
        return _tt2ps_map[K]
    else:
        fn, b1, i1 = ps2tt(K[0])
        K = (fn, b1 | b, i1 | i)
        if K in _tt2ps_map:
            return _tt2ps_map[K]
    raise ValueError("Can't find concrete font for family=%s, bold=%d, italic=%d" % (fn, b, i))
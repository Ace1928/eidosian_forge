def write_weave(weave, f, format=None):
    if format is None or format == 1:
        return write_weave_v5(weave, f)
    else:
        raise ValueError('unknown weave format %r' % format)
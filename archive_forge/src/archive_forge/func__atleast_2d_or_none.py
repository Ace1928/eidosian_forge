import cupy
def _atleast_2d_or_none(arg):
    if arg is not None:
        return cupy.atleast_2d(arg)
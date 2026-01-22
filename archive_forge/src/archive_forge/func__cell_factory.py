import sys
def _cell_factory():
    a = 1

    def f():
        nonlocal a
    return f.__closure__[0]
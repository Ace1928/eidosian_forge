from collections import namedtuple
def try_int(x):
    try:
        return int(x)
    except ValueError:
        return None
import re
def _prefer_non_zero(*args):
    for arg in args:
        if arg != 0:
            return arg
    return 0.0
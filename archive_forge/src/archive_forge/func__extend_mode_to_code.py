from collections.abc import Iterable
import operator
import warnings
import numpy
def _extend_mode_to_code(mode):
    """Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode in ['reflect', 'grid-mirror']:
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    elif mode == 'grid-wrap':
        return 5
    elif mode == 'grid-constant':
        return 6
    else:
        raise RuntimeError('boundary mode not supported')
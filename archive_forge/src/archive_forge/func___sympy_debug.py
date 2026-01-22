import sys
import re
def __sympy_debug():
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        raise RuntimeError('unrecognized value for SYMPY_DEBUG: %s' % debug_str)
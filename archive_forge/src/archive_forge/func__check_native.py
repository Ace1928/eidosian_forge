import re
import sys
from builtins import str, chr
def _check_native(tbl):
    """
    Determine if Python's own native implementation
    subsumes the supplied case folding table
    """
    try:
        for i in tbl:
            stv = chr(i)
            if stv.casefold() == stv:
                return False
    except AttributeError:
        return False
    return True
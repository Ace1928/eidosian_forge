import __future__
import warnings
def _is_syntax_error(err1, err2):
    rep1 = repr(err1)
    rep2 = repr(err2)
    if 'was never closed' in rep1 and 'was never closed' in rep2:
        return False
    if rep1 == rep2:
        return True
    return False
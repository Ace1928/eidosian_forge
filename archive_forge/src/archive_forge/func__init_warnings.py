import warnings
from cupyx import _ufunc_config
def _init_warnings():
    FallbackWarning = type('FallbackWarning', (Warning,), {})
    warnings.simplefilter(action='always', category=FallbackWarning)
    return FallbackWarning
import reprlib
from _thread import get_ident
from . import format_helpers
@reprlib.recursive_repr()
def _future_repr(future):
    info = ' '.join(_future_repr_info(future))
    return f'<{future.__class__.__name__} {info}>'
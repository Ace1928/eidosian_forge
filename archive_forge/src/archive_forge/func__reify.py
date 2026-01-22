from .core import unify, reify  # type: ignore[attr-defined]
from .dispatch import dispatch
@dispatch(slice, dict)
def _reify(o, s):
    """ Reify a Python ``slice`` object """
    return slice(*reify((o.start, o.stop, o.step), s))
from .core import unify, reify  # type: ignore[attr-defined]
from .dispatch import dispatch
def _reify_object_dict(o, s):
    obj = object.__new__(type(o))
    d = reify(o.__dict__, s)
    if d == o.__dict__:
        return o
    obj.__dict__.update(d)
    return obj
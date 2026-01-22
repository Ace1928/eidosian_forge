from __future__ import annotations
import sqlalchemy as sa
from .. import exc as sa_exc
from ..orm.writeonly import WriteOnlyCollection
class BasicEntity:

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)

    def __repr__(self):
        if id(self) in _repr_stack:
            return object.__repr__(self)
        _repr_stack.add(id(self))
        try:
            return '%s(%s)' % (self.__class__.__name__, ', '.join(['%s=%r' % (key, getattr(self, key)) for key in sorted(self.__dict__.keys()) if not key.startswith('_')]))
        finally:
            _repr_stack.remove(id(self))
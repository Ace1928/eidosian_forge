from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from .util import _mapper_property_as_plain_name
from .. import exc as sa_exc
from .. import util
from ..exc import MultipleResultsFound  # noqa
from ..exc import NoResultFound  # noqa
class UnmappedInstanceError(UnmappedError):
    """An mapping operation was requested for an unknown instance."""

    @util.preload_module('sqlalchemy.orm.base')
    def __init__(self, obj: object, msg: Optional[str]=None):
        base = util.preloaded.orm_base
        if not msg:
            try:
                base.class_mapper(type(obj))
                name = _safe_cls_name(type(obj))
                msg = 'Class %r is mapped, but this instance lacks instrumentation.  This occurs when the instance is created before sqlalchemy.orm.mapper(%s) was called.' % (name, name)
            except UnmappedClassError:
                msg = f"Class '{_safe_cls_name(type(obj))}' is not mapped"
                if isinstance(obj, type):
                    msg += '; was a class (%s) supplied where an instance was required?' % _safe_cls_name(obj)
        UnmappedError.__init__(self, msg)

    def __reduce__(self) -> Any:
        return (self.__class__, (None, self.args[0]))
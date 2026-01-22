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
@util.preload_module('sqlalchemy.orm.base')
def _default_unmapped(cls: Type[Any]) -> Optional[str]:
    base = util.preloaded.orm_base
    try:
        mappers = base.manager_of_class(cls).mappers
    except (UnmappedClassError, TypeError) + NO_STATE:
        mappers = {}
    name = _safe_cls_name(cls)
    if not mappers:
        return f"Class '{name}' is not mapped"
    else:
        return None
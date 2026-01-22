from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self
def _parse_attr_argument(attr: _AttrType) -> Tuple[InspectionAttr, _InternalEntityType[Any], MapperProperty[Any]]:
    """parse an attribute or wildcard argument to produce an
    :class:`._AbstractLoad` instance.

    This is used by the standalone loader strategy functions like
    ``joinedload()``, ``defer()``, etc. to produce :class:`_orm.Load` or
    :class:`._WildcardLoad` objects.

    """
    try:
        insp: InspectionAttr = inspect(attr)
    except sa_exc.NoInspectionAvailable as err:
        raise sa_exc.ArgumentError('expected ORM mapped attribute for loader strategy argument') from err
    lead_entity: _InternalEntityType[Any]
    if insp_is_mapper_property(insp):
        lead_entity = insp.parent
        prop = insp
    elif insp_is_attribute(insp):
        lead_entity = insp.parent
        prop = insp.prop
    else:
        raise sa_exc.ArgumentError('expected ORM mapped attribute for loader strategy argument')
    return (insp, lead_entity, prop)
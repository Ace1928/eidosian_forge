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
def _raise_for_does_not_link(path, attrname, parent_entity):
    if len(path) > 1:
        path_is_of_type = path[-1].entity is not path[-2].mapper.class_
        if insp_is_aliased_class(parent_entity):
            parent_entity_str = str(parent_entity)
        else:
            parent_entity_str = parent_entity.class_.__name__
        raise sa_exc.ArgumentError(f'ORM mapped entity or attribute "{attrname}" does not link from relationship "{path[-2]}%s".%s' % (f'.of_type({path[-1]})' if path_is_of_type else '', f'  Did you mean to use "{path[-2]}.of_type({parent_entity_str})" or "loadopt.options(selectin_polymorphic({path[-2].mapper.class_.__name__}, [{parent_entity_str}]), ...)" ?' if not path_is_of_type and (not path[-1].is_aliased_class) and orm_util._entity_corresponds_to(path.entity, inspect(parent_entity).mapper) else ''))
    else:
        raise sa_exc.ArgumentError(f'ORM mapped attribute "{attrname}" does not link mapped class "{path[-1]}"')
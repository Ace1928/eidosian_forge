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
@classmethod
def _chop_path(cls, to_chop: _PathRepresentation, path: PathRegistry, debug: bool=False) -> Optional[_PathRepresentation]:
    i = -1
    for i, (c_token, p_token) in enumerate(zip(to_chop, path.natural_path)):
        if isinstance(c_token, str):
            if i == 0 and (c_token.endswith(f':{_DEFAULT_TOKEN}') or c_token.endswith(f':{_WILDCARD_TOKEN}')):
                return to_chop
            elif c_token != f'{_RELATIONSHIP_TOKEN}:{_WILDCARD_TOKEN}' and c_token != p_token.key:
                return None
        if c_token is p_token:
            continue
        elif isinstance(c_token, InspectionAttr) and insp_is_mapper(c_token) and insp_is_mapper(p_token) and c_token.isa(p_token):
            continue
        else:
            return None
    return to_chop[i + 1:]
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
def _apply_to_parent(self, parent: Load) -> None:
    """apply this :class:`_orm._WildcardLoad` object as a sub-option of
        a :class:`_orm.Load` object.

        This method is used by the :meth:`_orm.Load.options` method.   Note
        that :class:`_orm.WildcardLoad` itself can't have sub-options, but
        it may be used as the sub-option of a :class:`_orm.Load` object.

        """
    assert self.path
    attr = self.path[0]
    if attr.endswith(_DEFAULT_TOKEN):
        attr = f'{attr.split(':')[0]}:{_WILDCARD_TOKEN}'
    effective_path = cast(AbstractEntityRegistry, parent.path).token(attr)
    assert effective_path.is_token
    loader = _TokenStrategyLoad.create(effective_path, None, self.strategy, None, self.local_opts, self.propagate_to_loaders)
    parent.context += (loader,)
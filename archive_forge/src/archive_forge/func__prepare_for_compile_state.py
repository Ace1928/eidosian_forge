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
def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
    current_path = compile_state.current_path
    is_refresh = compile_state.compile_options._for_refresh_state
    if is_refresh and (not self.propagate_to_loaders):
        return []
    if not self.strategy and (not self.local_opts):
        return []
    effective_path = self.path
    if current_path:
        new_effective_path = self._adjust_effective_path_for_current_path(effective_path, current_path)
        if new_effective_path is None:
            return []
        effective_path = new_effective_path
    return [('loader', effective_path.natural_path)]
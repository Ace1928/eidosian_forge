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
class _WildcardLoad(_AbstractLoad):
    """represent a standalone '*' load operation"""
    __slots__ = ('strategy', 'path', 'local_opts')
    _traverse_internals = [('strategy', visitors.ExtendedInternalTraversal.dp_plain_obj), ('path', visitors.ExtendedInternalTraversal.dp_plain_obj), ('local_opts', visitors.ExtendedInternalTraversal.dp_string_multi_dict)]
    cache_key_traversal: _CacheKeyTraversalType = None
    strategy: Optional[Tuple[Any, ...]]
    local_opts: _OptsType
    path: Union[Tuple[()], Tuple[str]]
    propagate_to_loaders = False

    def __init__(self) -> None:
        self.path = ()
        self.strategy = None
        self.local_opts = util.EMPTY_DICT

    def _clone_for_bind_strategy(self, attrs, strategy, wildcard_key, opts=None, attr_group=None, propagate_to_loaders=True, reconcile_to_other=None, extra_criteria=None):
        assert attrs is not None
        attr = attrs[0]
        assert wildcard_key and isinstance(attr, str) and (attr in (_WILDCARD_TOKEN, _DEFAULT_TOKEN))
        attr = f'{wildcard_key}:{attr}'
        self.strategy = strategy
        self.path = (attr,)
        if opts:
            self.local_opts = util.immutabledict(opts)
        assert extra_criteria is None

    def options(self, *opts: _AbstractLoad) -> Self:
        raise NotImplementedError('Star option does not support sub-options')

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

    def _process(self, compile_state, mapper_entities, raiseerr):
        is_refresh = compile_state.compile_options._for_refresh_state
        if is_refresh and (not self.propagate_to_loaders):
            return
        entities = [ent.entity_zero for ent in mapper_entities]
        current_path = compile_state.current_path
        start_path: _PathRepresentation = self.path
        if current_path:
            new_path = self._chop_path(start_path, current_path)
            if new_path is None:
                return
            assert new_path == start_path
        assert start_path and len(start_path) == 1
        token = start_path[0]
        assert isinstance(token, str)
        entity = self._find_entity_basestring(entities, token, raiseerr)
        if not entity:
            return
        path_element = entity
        assert isinstance(token, str)
        loader = _TokenStrategyLoad.create(path_element._path_registry, token, self.strategy, None, self.local_opts, self.propagate_to_loaders, raiseerr=raiseerr)
        if not loader:
            return
        assert loader.path.is_token
        loader.process_compile_state(self, compile_state, mapper_entities, None, raiseerr)
        return loader

    def _find_entity_basestring(self, entities: Iterable[_InternalEntityType[Any]], token: str, raiseerr: bool) -> Optional[_InternalEntityType[Any]]:
        if token.endswith(f':{_WILDCARD_TOKEN}'):
            if len(list(entities)) != 1:
                if raiseerr:
                    raise sa_exc.ArgumentError(f"Can't apply wildcard ('*') or load_only() loader option to multiple entities {', '.join((str(ent) for ent in entities))}. Specify loader options for each entity individually, such as {', '.join((f"Load({ent}).some_option('*')" for ent in entities))}.")
        elif token.endswith(_DEFAULT_TOKEN):
            raiseerr = False
        for ent in entities:
            return ent
        else:
            if raiseerr:
                raise sa_exc.ArgumentError(f'''Query has only expression-based entities - can't find property named "{token}".''')
            else:
                return None

    def __getstate__(self) -> Dict[str, Any]:
        d = self._shallow_to_dict()
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._shallow_from_dict(state)
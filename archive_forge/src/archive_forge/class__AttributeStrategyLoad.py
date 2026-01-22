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
class _AttributeStrategyLoad(_LoadElement):
    """Loader strategies against specific relationship or column paths.

    e.g.::

        joinedload(User.addresses)
        defer(Order.name)
        selectinload(User.orders).lazyload(Order.items)

    """
    __slots__ = ('_of_type', '_path_with_polymorphic_path')
    __visit_name__ = 'attribute_strategy_load_element'
    _traverse_internals = _LoadElement._traverse_internals + [('_of_type', visitors.ExtendedInternalTraversal.dp_multi), ('_path_with_polymorphic_path', visitors.ExtendedInternalTraversal.dp_has_cache_key)]
    _of_type: Union[Mapper[Any], AliasedInsp[Any], None]
    _path_with_polymorphic_path: Optional[PathRegistry]
    is_class_strategy = False
    is_token_strategy = False

    def _init_path(self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria):
        assert attr is not None
        self._of_type = None
        self._path_with_polymorphic_path = None
        insp, _, prop = _parse_attr_argument(attr)
        if insp.is_property:
            prop = attr
            path = path[prop]
            if path.has_entity:
                path = path.entity_path
            return path
        elif not insp.is_attribute:
            assert False
        if not orm_util._entity_corresponds_to_use_path_impl(path[-1], attr.parent):
            if raiseerr:
                if attr_group and attr is not attr_group[0]:
                    raise sa_exc.ArgumentError("Can't apply wildcard ('*') or load_only() loader option to multiple entities in the same option. Use separate options per entity.")
                else:
                    _raise_for_does_not_link(path, str(attr), attr.parent)
            else:
                return None
        if extra_criteria:
            assert not attr._extra_criteria
            self._extra_criteria = extra_criteria
        else:
            self._extra_criteria = attr._extra_criteria
        if getattr(attr, '_of_type', None):
            ac = attr._of_type
            ext_info = inspect(ac)
            self._of_type = ext_info
            self._path_with_polymorphic_path = path.entity_path[prop]
            path = path[prop][ext_info]
        else:
            path = path[prop]
        if path.has_entity:
            path = path.entity_path
        return path

    def _generate_extra_criteria(self, context):
        """Apply the current bound parameters in a QueryContext to the
        immediate "extra_criteria" stored with this Load object.

        Load objects are typically pulled from the cached version of
        the statement from a QueryContext.  The statement currently being
        executed will have new values (and keys) for bound parameters in the
        extra criteria which need to be applied by loader strategies when
        they handle this criteria for a result set.

        """
        assert self._extra_criteria, 'this should only be called if _extra_criteria is present'
        orig_query = context.compile_state.select_statement
        current_query = context.query
        k1 = orig_query._generate_cache_key()
        k2 = current_query._generate_cache_key()
        return k2._apply_params_to_element(k1, and_(*self._extra_criteria))

    def _set_of_type_info(self, context, current_path):
        assert self._path_with_polymorphic_path
        pwpi = self._of_type
        assert pwpi
        if not pwpi.is_aliased_class:
            pwpi = inspect(orm_util.AliasedInsp._with_polymorphic_factory(pwpi.mapper.base_mapper, (pwpi.mapper,), aliased=True, _use_mapper_path=True))
        start_path = self._path_with_polymorphic_path
        if current_path:
            new_path = self._adjust_effective_path_for_current_path(start_path, current_path)
            if new_path is None:
                return
            start_path = new_path
        key = ('path_with_polymorphic', start_path.natural_path)
        if key in context:
            existing_aliased_insp = context[key]
            this_aliased_insp = pwpi
            new_aliased_insp = existing_aliased_insp._merge_with(this_aliased_insp)
            context[key] = new_aliased_insp
        else:
            context[key] = pwpi

    def _prepare_for_compile_state(self, parent_loader, compile_state, mapper_entities, reconciled_lead_entity, raiseerr):
        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state
        assert not self.path.is_token
        if is_refresh and (not self.propagate_to_loaders):
            return []
        if self._of_type:
            self._set_of_type_info(compile_state.attributes, current_path)
        if not self.strategy and (not self.local_opts):
            return []
        if raiseerr and (not reconciled_lead_entity):
            self._raise_for_no_match(parent_loader, mapper_entities)
        if self.path.has_entity:
            effective_path = self.path.parent
        else:
            effective_path = self.path
        if current_path:
            assert effective_path is not None
            effective_path = self._adjust_effective_path_for_current_path(effective_path, current_path)
            if effective_path is None:
                return []
        return [('loader', cast(PathRegistry, effective_path).natural_path)]

    def __getstate__(self):
        d = super().__getstate__()
        d['_extra_criteria'] = ()
        if self._path_with_polymorphic_path:
            d['_path_with_polymorphic_path'] = self._path_with_polymorphic_path.serialize()
        if self._of_type:
            if self._of_type.is_aliased_class:
                d['_of_type'] = None
            elif self._of_type.is_mapper:
                d['_of_type'] = self._of_type.class_
            else:
                assert False, 'unexpected object for _of_type'
        return d

    def __setstate__(self, state):
        super().__setstate__(state)
        if state.get('_path_with_polymorphic_path', None):
            self._path_with_polymorphic_path = PathRegistry.deserialize(state['_path_with_polymorphic_path'])
        else:
            self._path_with_polymorphic_path = None
        if state.get('_of_type', None):
            self._of_type = inspect(state['_of_type'])
        else:
            self._of_type = None
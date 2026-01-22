from __future__ import annotations
from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
class TokenRegistry(PathRegistry):
    __slots__ = ('token', 'parent', 'path', 'natural_path')
    inherit_cache = True
    token: _StrPathToken
    parent: CreatesToken

    def __init__(self, parent: CreatesToken, token: _StrPathToken):
        token = PathToken.intern(token)
        self.token = token
        self.parent = parent
        self.path = parent.path + (token,)
        self.natural_path = parent.natural_path + (token,)
    has_entity = False
    is_token = True

    def generate_for_superclasses(self) -> Iterator[PathRegistry]:
        parent = self.parent
        if is_root(parent):
            yield self
            return
        if TYPE_CHECKING:
            assert isinstance(parent, AbstractEntityRegistry)
        if not parent.is_aliased_class:
            for mp_ent in parent.mapper.iterate_to_root():
                yield TokenRegistry(parent.parent[mp_ent], self.token)
        elif parent.is_aliased_class and cast('AliasedInsp[Any]', parent.entity)._is_with_polymorphic:
            yield self
            for ent in cast('AliasedInsp[Any]', parent.entity)._with_polymorphic_entities:
                yield TokenRegistry(parent.parent[ent], self.token)
        else:
            yield self

    def _generate_natural_for_superclasses(self) -> Iterator[_PathRepresentation]:
        parent = self.parent
        if is_root(parent):
            yield self.natural_path
            return
        if TYPE_CHECKING:
            assert isinstance(parent, AbstractEntityRegistry)
        for mp_ent in parent.mapper.iterate_to_root():
            yield TokenRegistry(parent.parent[mp_ent], self.token).natural_path
        if parent.is_aliased_class and cast('AliasedInsp[Any]', parent.entity)._is_with_polymorphic:
            yield self.natural_path
            for ent in cast('AliasedInsp[Any]', parent.entity)._with_polymorphic_entities:
                yield TokenRegistry(parent.parent[ent], self.token).natural_path
        else:
            yield self.natural_path

    def _getitem(self, entity: Any) -> Any:
        try:
            return self.path[entity]
        except TypeError as err:
            raise IndexError(f'{entity}') from err
    if not TYPE_CHECKING:
        __getitem__ = _getitem
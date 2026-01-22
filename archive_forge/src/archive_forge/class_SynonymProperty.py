from __future__ import annotations
from dataclasses import is_dataclass
import inspect
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import util as orm_util
from .base import _DeclarativeMapped
from .base import LoaderCallableStatus
from .base import Mapped
from .base import PassiveFlag
from .base import SQLORMOperations
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .util import _none_set
from .util import de_stringify_annotation
from .. import event
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import util
from ..sql import expression
from ..sql import operators
from ..sql.elements import BindParameter
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593
from ..util.typing import typing_get_args
class SynonymProperty(DescriptorProperty[_T]):
    """Denote an attribute name as a synonym to a mapped property,
    in that the attribute will mirror the value and expression behavior
    of another attribute.

    :class:`.Synonym` is constructed using the :func:`_orm.synonym`
    function.

    .. seealso::

        :ref:`synonyms` - Overview of synonyms

    """
    comparator_factory: Optional[Type[PropComparator[_T]]]

    def __init__(self, name: str, map_column: Optional[bool]=None, descriptor: Optional[Any]=None, comparator_factory: Optional[Type[PropComparator[_T]]]=None, attribute_options: Optional[_AttributeOptions]=None, info: Optional[_InfoType]=None, doc: Optional[str]=None):
        super().__init__(attribute_options=attribute_options)
        self.name = name
        self.map_column = map_column
        self.descriptor = descriptor
        self.comparator_factory = comparator_factory
        if doc:
            self.doc = doc
        elif descriptor and descriptor.__doc__:
            self.doc = descriptor.__doc__
        else:
            self.doc = None
        if info:
            self.info.update(info)
        util.set_creation_order(self)
    if not TYPE_CHECKING:

        @property
        def uses_objects(self) -> bool:
            return getattr(self.parent.class_, self.name).impl.uses_objects

    @util.memoized_property
    def _proxied_object(self) -> Union[MapperProperty[_T], SQLORMOperations[_T]]:
        attr = getattr(self.parent.class_, self.name)
        if not hasattr(attr, 'property') or not isinstance(attr.property, MapperProperty):
            if isinstance(attr, attributes.QueryableAttribute):
                return attr.comparator
            elif isinstance(attr, SQLORMOperations):
                return attr
            raise sa_exc.InvalidRequestError('synonym() attribute "%s.%s" only supports ORM mapped attributes, got %r' % (self.parent.class_.__name__, self.name, attr))
        return attr.property

    def _comparator_factory(self, mapper: Mapper[Any]) -> SQLORMOperations[_T]:
        prop = self._proxied_object
        if isinstance(prop, MapperProperty):
            if self.comparator_factory:
                comp = self.comparator_factory(prop, mapper)
            else:
                comp = prop.comparator_factory(prop, mapper)
            return comp
        else:
            return prop

    def get_history(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF) -> History:
        attr: QueryableAttribute[Any] = getattr(self.parent.class_, self.name)
        return attr.impl.get_history(state, dict_, passive=passive)

    @util.preload_module('sqlalchemy.orm.properties')
    def set_parent(self, parent: Mapper[Any], init: bool) -> None:
        properties = util.preloaded.orm_properties
        if self.map_column:
            if self.key not in parent.persist_selectable.c:
                raise sa_exc.ArgumentError("Can't compile synonym '%s': no column on table '%s' named '%s'" % (self.name, parent.persist_selectable.description, self.key))
            elif parent.persist_selectable.c[self.key] in parent._columntoproperty and parent._columntoproperty[parent.persist_selectable.c[self.key]].key == self.name:
                raise sa_exc.ArgumentError("Can't call map_column=True for synonym %r=%r, a ColumnProperty already exists keyed to the name %r for column %r" % (self.key, self.name, self.name, self.key))
            p: ColumnProperty[Any] = properties.ColumnProperty(parent.persist_selectable.c[self.key])
            parent._configure_property(self.name, p, init=init, setparent=True)
            p._mapped_by_synonym = self.key
        self.parent = parent
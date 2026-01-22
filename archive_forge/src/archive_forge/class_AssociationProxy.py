from __future__ import annotations
import operator
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import ItemsView
from typing import Iterable
from typing import Iterator
from typing import KeysView
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import MutableSet
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import ValuesView
from .. import ColumnElement
from .. import exc
from .. import inspect
from .. import orm
from .. import util
from ..orm import collections
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.base import SQLORMOperations
from ..orm.interfaces import _AttributeOptions
from ..orm.interfaces import _DCAttributeOptions
from ..orm.interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from ..sql import operators
from ..sql import or_
from ..sql.base import _NoArg
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import SupportsIndex
from ..util.typing import SupportsKeysAndGetItem
class AssociationProxy(interfaces.InspectionAttrInfo, ORMDescriptor[_T], _DCAttributeOptions, _AssociationProxyProtocol[_T]):
    """A descriptor that presents a read/write view of an object attribute."""
    is_attribute = True
    extension_type = AssociationProxyExtensionType.ASSOCIATION_PROXY

    def __init__(self, target_collection: str, attr: str, *, creator: Optional[_CreatorProtocol]=None, getset_factory: Optional[_GetSetFactoryProtocol]=None, proxy_factory: Optional[_ProxyFactoryProtocol]=None, proxy_bulk_set: Optional[_ProxyBulkSetProtocol]=None, info: Optional[_InfoType]=None, cascade_scalar_deletes: bool=False, create_on_none_assignment: bool=False, attribute_options: Optional[_AttributeOptions]=None):
        """Construct a new :class:`.AssociationProxy`.

        The :class:`.AssociationProxy` object is typically constructed using
        the :func:`.association_proxy` constructor function. See the
        description of :func:`.association_proxy` for a description of all
        parameters.


        """
        self.target_collection = target_collection
        self.value_attr = attr
        self.creator = creator
        self.getset_factory = getset_factory
        self.proxy_factory = proxy_factory
        self.proxy_bulk_set = proxy_bulk_set
        if cascade_scalar_deletes and create_on_none_assignment:
            raise exc.ArgumentError('The cascade_scalar_deletes and create_on_none_assignment parameters are mutually exclusive.')
        self.cascade_scalar_deletes = cascade_scalar_deletes
        self.create_on_none_assignment = create_on_none_assignment
        self.key = '_%s_%s_%s' % (type(self).__name__, target_collection, id(self))
        if info:
            self.info = info
        if attribute_options and attribute_options != _DEFAULT_ATTRIBUTE_OPTIONS:
            self._has_dataclass_arguments = True
            self._attribute_options = attribute_options
        else:
            self._has_dataclass_arguments = False
            self._attribute_options = _DEFAULT_ATTRIBUTE_OPTIONS

    @overload
    def __get__(self, instance: Literal[None], owner: Literal[None]) -> Self:
        ...

    @overload
    def __get__(self, instance: Literal[None], owner: Any) -> AssociationProxyInstance[_T]:
        ...

    @overload
    def __get__(self, instance: object, owner: Any) -> _T:
        ...

    def __get__(self, instance: object, owner: Any) -> Union[AssociationProxyInstance[_T], _T, AssociationProxy[_T]]:
        if owner is None:
            return self
        inst = self._as_instance(owner, instance)
        if inst:
            return inst.get(instance)
        assert instance is None
        return self

    def __set__(self, instance: object, values: _T) -> None:
        class_ = type(instance)
        self._as_instance(class_, instance).set(instance, values)

    def __delete__(self, instance: object) -> None:
        class_ = type(instance)
        self._as_instance(class_, instance).delete(instance)

    def for_class(self, class_: Type[Any], obj: Optional[object]=None) -> AssociationProxyInstance[_T]:
        """Return the internal state local to a specific mapped class.

        E.g., given a class ``User``::

            class User(Base):
                # ...

                keywords = association_proxy('kws', 'keyword')

        If we access this :class:`.AssociationProxy` from
        :attr:`_orm.Mapper.all_orm_descriptors`, and we want to view the
        target class for this proxy as mapped by ``User``::

            inspect(User).all_orm_descriptors["keywords"].for_class(User).target_class

        This returns an instance of :class:`.AssociationProxyInstance` that
        is specific to the ``User`` class.   The :class:`.AssociationProxy`
        object remains agnostic of its parent class.

        :param class\\_: the class that we are returning state for.

        :param obj: optional, an instance of the class that is required
         if the attribute refers to a polymorphic target, e.g. where we have
         to look at the type of the actual destination object to get the
         complete path.

        .. versionadded:: 1.3 - :class:`.AssociationProxy` no longer stores
           any state specific to a particular parent class; the state is now
           stored in per-class :class:`.AssociationProxyInstance` objects.


        """
        return self._as_instance(class_, obj)

    def _as_instance(self, class_: Any, obj: Any) -> AssociationProxyInstance[_T]:
        try:
            inst = class_.__dict__[self.key + '_inst']
        except KeyError:
            inst = None
        if inst is None:
            owner = self._calc_owner(class_)
            if owner is not None:
                inst = AssociationProxyInstance.for_proxy(self, owner, obj)
                setattr(class_, self.key + '_inst', inst)
            else:
                inst = None
        if inst is not None and (not inst._is_canonical):
            return inst._non_canonical_get_for_object(obj)
        else:
            return inst

    def _calc_owner(self, target_cls: Any) -> Any:
        try:
            insp = inspect(target_cls)
        except exc.NoInspectionAvailable:
            return None
        else:
            return insp.mapper.class_manager.class_

    def _default_getset(self, collection_class: Any) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
        attr = self.value_attr
        _getter = operator.attrgetter(attr)

        def getter(instance: Any) -> Optional[Any]:
            return _getter(instance) if instance is not None else None
        if collection_class is dict:

            def dict_setter(instance: Any, k: Any, value: Any) -> None:
                setattr(instance, attr, value)
            return (getter, dict_setter)
        else:

            def plain_setter(o: Any, v: Any) -> None:
                setattr(o, attr, v)
            return (getter, plain_setter)

    def __repr__(self) -> str:
        return 'AssociationProxy(%r, %r)' % (self.target_collection, self.value_attr)
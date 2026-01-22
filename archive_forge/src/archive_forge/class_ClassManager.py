from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import base
from . import collections
from . import exc
from . import interfaces
from . import state
from ._typing import _O
from .attributes import _is_collection_attribute_impl
from .. import util
from ..event import EventTarget
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
class ClassManager(HasMemoized, Dict[str, 'QueryableAttribute[Any]'], Generic[_O], EventTarget):
    """Tracks state information at the class level."""
    dispatch: dispatcher[ClassManager[_O]]
    MANAGER_ATTR = base.DEFAULT_MANAGER_ATTR
    STATE_ATTR = base.DEFAULT_STATE_ATTR
    _state_setter = staticmethod(util.attrsetter(STATE_ATTR))
    expired_attribute_loader: _ExpiredAttributeLoaderProto
    'previously known as deferred_scalar_loader'
    init_method: Optional[Callable[..., None]]
    original_init: Optional[Callable[..., None]] = None
    factory: Optional[_ManagerFactory]
    declarative_scan: Optional[weakref.ref[_MapperConfig]] = None
    registry: _RegistryType
    if not TYPE_CHECKING:
        registry = None
    class_: Type[_O]
    _bases: List[ClassManager[Any]]

    @property
    @util.deprecated('1.4', message='The ClassManager.deferred_scalar_loader attribute is now named expired_attribute_loader')
    def deferred_scalar_loader(self):
        return self.expired_attribute_loader

    @deferred_scalar_loader.setter
    @util.deprecated('1.4', message='The ClassManager.deferred_scalar_loader attribute is now named expired_attribute_loader')
    def deferred_scalar_loader(self, obj):
        self.expired_attribute_loader = obj

    def __init__(self, class_):
        self.class_ = class_
        self.info = {}
        self.new_init = None
        self.local_attrs = {}
        self.originals = {}
        self._finalized = False
        self.factory = None
        self.init_method = None
        self._bases = [mgr for mgr in cast('List[Optional[ClassManager[Any]]]', [opt_manager_of_class(base) for base in self.class_.__bases__ if isinstance(base, type)]) if mgr is not None]
        for base_ in self._bases:
            self.update(base_)
        cast('InstanceEvents', self.dispatch._events)._new_classmanager_instance(class_, self)
        for basecls in class_.__mro__:
            mgr = opt_manager_of_class(basecls)
            if mgr is not None:
                self.dispatch._update(mgr.dispatch)
        self.manage()
        if '__del__' in class_.__dict__:
            util.warn('__del__() method on class %s will cause unreachable cycles and memory leaks, as SQLAlchemy instrumentation often creates reference cycles.  Please remove this method.' % class_)

    def _update_state(self, finalize: bool=False, mapper: Optional[Mapper[_O]]=None, registry: Optional[_RegistryType]=None, declarative_scan: Optional[_MapperConfig]=None, expired_attribute_loader: Optional[_ExpiredAttributeLoaderProto]=None, init_method: Optional[Callable[..., None]]=None) -> None:
        if mapper:
            self.mapper = mapper
        if registry:
            registry._add_manager(self)
        if declarative_scan:
            self.declarative_scan = weakref.ref(declarative_scan)
        if expired_attribute_loader:
            self.expired_attribute_loader = expired_attribute_loader
        if init_method:
            assert not self._finalized, "class is already instrumented, init_method %s can't be applied" % init_method
            self.init_method = init_method
        if not self._finalized:
            self.original_init = self.init_method if self.init_method is not None and self.class_.__init__ is object.__init__ else self.class_.__init__
        if finalize and (not self._finalized):
            self._finalize()

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._instrument_init()
        _instrumentation_factory.dispatch.class_instrument(self.class_)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return other is self

    @property
    def is_mapped(self) -> bool:
        return 'mapper' in self.__dict__

    @HasMemoized.memoized_attribute
    def _all_key_set(self):
        return frozenset(self)

    @HasMemoized.memoized_attribute
    def _collection_impl_keys(self):
        return frozenset([attr.key for attr in self.values() if attr.impl.collection])

    @HasMemoized.memoized_attribute
    def _scalar_loader_impls(self):
        return frozenset([attr.impl for attr in self.values() if attr.impl.accepts_scalar_loader])

    @HasMemoized.memoized_attribute
    def _loader_impls(self):
        return frozenset([attr.impl for attr in self.values()])

    @util.memoized_property
    def mapper(self) -> Mapper[_O]:
        raise exc.UnmappedClassError(self.class_)

    def _all_sqla_attributes(self, exclude=None):
        """return an iterator of all classbound attributes that are
        implement :class:`.InspectionAttr`.

        This includes :class:`.QueryableAttribute` as well as extension
        types such as :class:`.hybrid_property` and
        :class:`.AssociationProxy`.

        """
        found: Dict[str, Any] = {}
        for supercls in self.class_.__mro__[0:-1]:
            inherits = supercls.__mro__[1]
            for key in supercls.__dict__:
                found.setdefault(key, supercls)
                if key in inherits.__dict__:
                    continue
                val = found[key].__dict__[key]
                if isinstance(val, interfaces.InspectionAttr) and val.is_attribute:
                    yield (key, val)

    def _get_class_attr_mro(self, key, default=None):
        """return an attribute on the class without tripping it."""
        for supercls in self.class_.__mro__:
            if key in supercls.__dict__:
                return supercls.__dict__[key]
        else:
            return default

    def _attr_has_impl(self, key: str) -> bool:
        """Return True if the given attribute is fully initialized.

        i.e. has an impl.
        """
        return key in self and self[key].impl is not None

    def _subclass_manager(self, cls: Type[_T]) -> ClassManager[_T]:
        """Create a new ClassManager for a subclass of this ClassManager's
        class.

        This is called automatically when attributes are instrumented so that
        the attributes can be propagated to subclasses against their own
        class-local manager, without the need for mappers etc. to have already
        pre-configured managers for the full class hierarchy.   Mappers
        can post-configure the auto-generated ClassManager when needed.

        """
        return register_class(cls, finalize=False)

    def _instrument_init(self):
        self.new_init = _generate_init(self.class_, self, self.original_init)
        self.install_member('__init__', self.new_init)

    @util.memoized_property
    def _state_constructor(self) -> Type[state.InstanceState[_O]]:
        self.dispatch.first_init(self, self.class_)
        return state.InstanceState

    def manage(self):
        """Mark this instance as the manager for its class."""
        setattr(self.class_, self.MANAGER_ATTR, self)

    @util.hybridmethod
    def manager_getter(self):
        return _default_manager_getter

    @util.hybridmethod
    def state_getter(self):
        """Return a (instance) -> InstanceState callable.

        "state getter" callables should raise either KeyError or
        AttributeError if no InstanceState could be found for the
        instance.
        """
        return _default_state_getter

    @util.hybridmethod
    def dict_getter(self):
        return _default_dict_getter

    def instrument_attribute(self, key: str, inst: QueryableAttribute[Any], propagated: bool=False) -> None:
        if propagated:
            if key in self.local_attrs:
                return
        else:
            self.local_attrs[key] = inst
            self.install_descriptor(key, inst)
        self._reset_memoizations()
        self[key] = inst
        for cls in self.class_.__subclasses__():
            manager = self._subclass_manager(cls)
            manager.instrument_attribute(key, inst, True)

    def subclass_managers(self, recursive):
        for cls in self.class_.__subclasses__():
            mgr = opt_manager_of_class(cls)
            if mgr is not None and mgr is not self:
                yield mgr
                if recursive:
                    yield from mgr.subclass_managers(True)

    def post_configure_attribute(self, key):
        _instrumentation_factory.dispatch.attribute_instrument(self.class_, key, self[key])

    def uninstrument_attribute(self, key, propagated=False):
        if key not in self:
            return
        if propagated:
            if key in self.local_attrs:
                return
        else:
            del self.local_attrs[key]
            self.uninstall_descriptor(key)
        self._reset_memoizations()
        del self[key]
        for cls in self.class_.__subclasses__():
            manager = opt_manager_of_class(cls)
            if manager:
                manager.uninstrument_attribute(key, True)

    def unregister(self) -> None:
        """remove all instrumentation established by this ClassManager."""
        for key in list(self.originals):
            self.uninstall_member(key)
        self.mapper = None
        self.dispatch = None
        self.new_init = None
        self.info.clear()
        for key in list(self):
            if key in self.local_attrs:
                self.uninstrument_attribute(key)
        if self.MANAGER_ATTR in self.class_.__dict__:
            delattr(self.class_, self.MANAGER_ATTR)

    def install_descriptor(self, key: str, inst: QueryableAttribute[Any]) -> None:
        if key in (self.STATE_ATTR, self.MANAGER_ATTR):
            raise KeyError('%r: requested attribute name conflicts with instrumentation attribute of the same name.' % key)
        setattr(self.class_, key, inst)

    def uninstall_descriptor(self, key: str) -> None:
        delattr(self.class_, key)

    def install_member(self, key: str, implementation: Any) -> None:
        if key in (self.STATE_ATTR, self.MANAGER_ATTR):
            raise KeyError('%r: requested attribute name conflicts with instrumentation attribute of the same name.' % key)
        self.originals.setdefault(key, self.class_.__dict__.get(key, DEL_ATTR))
        setattr(self.class_, key, implementation)

    def uninstall_member(self, key: str) -> None:
        original = self.originals.pop(key, None)
        if original is not DEL_ATTR:
            setattr(self.class_, key, original)
        else:
            delattr(self.class_, key)

    def instrument_collection_class(self, key: str, collection_class: Type[Collection[Any]]) -> _CollectionFactoryType:
        return collections.prepare_instrumentation(collection_class)

    def initialize_collection(self, key: str, state: InstanceState[_O], factory: _CollectionFactoryType) -> Tuple[collections.CollectionAdapter, _AdaptedCollectionProtocol]:
        user_data = factory()
        impl = self.get_impl(key)
        assert _is_collection_attribute_impl(impl)
        adapter = collections.CollectionAdapter(impl, state, user_data)
        return (adapter, user_data)

    def is_instrumented(self, key: str, search: bool=False) -> bool:
        if search:
            return key in self
        else:
            return key in self.local_attrs

    def get_impl(self, key: str) -> AttributeImpl:
        return self[key].impl

    @property
    def attributes(self) -> Iterable[Any]:
        return iter(self.values())

    def new_instance(self, state: Optional[InstanceState[_O]]=None) -> _O:
        instance = self.class_.__new__(self.class_)
        if state is None:
            state = self._state_constructor(instance, self)
        self._state_setter(instance, state)
        return instance

    def setup_instance(self, instance: _O, state: Optional[InstanceState[_O]]=None) -> None:
        if state is None:
            state = self._state_constructor(instance, self)
        self._state_setter(instance, state)

    def teardown_instance(self, instance: _O) -> None:
        delattr(instance, self.STATE_ATTR)

    def _serialize(self, state: InstanceState[_O], state_dict: Dict[str, Any]) -> _SerializeManager:
        return _SerializeManager(state, state_dict)

    def _new_state_if_none(self, instance: _O) -> Union[Literal[False], InstanceState[_O]]:
        """Install a default InstanceState if none is present.

        A private convenience method used by the __init__ decorator.

        """
        if hasattr(instance, self.STATE_ATTR):
            return False
        elif self.class_ is not instance.__class__ and self.is_mapped:
            return self._subclass_manager(instance.__class__)._new_state_if_none(instance)
        else:
            state = self._state_constructor(instance, self)
            self._state_setter(instance, state)
            return state

    def has_state(self, instance: _O) -> bool:
        return hasattr(instance, self.STATE_ATTR)

    def has_parent(self, state: InstanceState[_O], key: str, optimistic: bool=False) -> bool:
        """TODO"""
        return self.get_impl(key).hasparent(state, optimistic=optimistic)

    def __bool__(self) -> bool:
        """All ClassManagers are non-zero regardless of attribute state."""
        return True

    def __repr__(self) -> str:
        return '<%s of %r at %x>' % (self.__class__.__name__, self.class_, id(self))
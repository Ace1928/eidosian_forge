from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import attributes
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.attributes import QueryableAttribute
from ..sql import roles
from ..sql._typing import is_has_clause_element
from ..sql.elements import ColumnElement
from ..sql.elements import SQLCoreOperations
from ..util.typing import Concatenate
from ..util.typing import Literal
from ..util.typing import ParamSpec
from ..util.typing import Protocol
from ..util.typing import Self
class hybrid_property(interfaces.InspectionAttrInfo, ORMDescriptor[_T]):
    """A decorator which allows definition of a Python descriptor with both
    instance-level and class-level behavior.

    """
    is_attribute = True
    extension_type = HybridExtensionType.HYBRID_PROPERTY
    __name__: str

    def __init__(self, fget: _HybridGetterType[_T], fset: Optional[_HybridSetterType[_T]]=None, fdel: Optional[_HybridDeleterType[_T]]=None, expr: Optional[_HybridExprCallableType[_T]]=None, custom_comparator: Optional[Comparator[_T]]=None, update_expr: Optional[_HybridUpdaterType[_T]]=None):
        """Create a new :class:`.hybrid_property`.

        Usage is typically via decorator::

            from sqlalchemy.ext.hybrid import hybrid_property

            class SomeClass:
                @hybrid_property
                def value(self):
                    return self._value

                @value.setter
                def value(self, value):
                    self._value = value

        """
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.expr = _unwrap_classmethod(expr)
        self.custom_comparator = _unwrap_classmethod(custom_comparator)
        self.update_expr = _unwrap_classmethod(update_expr)
        util.update_wrapper(self, fget)

    @overload
    def __get__(self, instance: Any, owner: Literal[None]) -> Self:
        ...

    @overload
    def __get__(self, instance: Literal[None], owner: Type[object]) -> _HybridClassLevelAccessor[_T]:
        ...

    @overload
    def __get__(self, instance: object, owner: Type[object]) -> _T:
        ...

    def __get__(self, instance: Optional[object], owner: Optional[Type[object]]) -> Union[hybrid_property[_T], _HybridClassLevelAccessor[_T], _T]:
        if owner is None:
            return self
        elif instance is None:
            return self._expr_comparator(owner)
        else:
            return self.fget(instance)

    def __set__(self, instance: object, value: Any) -> None:
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(instance, value)

    def __delete__(self, instance: object) -> None:
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(instance)

    def _copy(self, **kw: Any) -> hybrid_property[_T]:
        defaults = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        defaults.update(**kw)
        return type(self)(**defaults)

    @property
    def overrides(self) -> Self:
        """Prefix for a method that is overriding an existing attribute.

        The :attr:`.hybrid_property.overrides` accessor just returns
        this hybrid object, which when called at the class level from
        a parent class, will de-reference the "instrumented attribute"
        normally returned at this level, and allow modifying decorators
        like :meth:`.hybrid_property.expression` and
        :meth:`.hybrid_property.comparator`
        to be used without conflicting with the same-named attributes
        normally present on the :class:`.QueryableAttribute`::

            class SuperClass:
                # ...

                @hybrid_property
                def foobar(self):
                    return self._foobar

            class SubClass(SuperClass):
                # ...

                @SuperClass.foobar.overrides.expression
                def foobar(cls):
                    return func.subfoobar(self._foobar)

        .. versionadded:: 1.2

        .. seealso::

            :ref:`hybrid_reuse_subclass`

        """
        return self

    class _InPlace(Generic[_TE]):
        """A builder helper for .hybrid_property.

        .. versionadded:: 2.0.4

        """
        __slots__ = ('attr',)

        def __init__(self, attr: hybrid_property[_TE]):
            self.attr = attr

        def _set(self, **kw: Any) -> hybrid_property[_TE]:
            for k, v in kw.items():
                setattr(self.attr, k, _unwrap_classmethod(v))
            return self.attr

        def getter(self, fget: _HybridGetterType[_TE]) -> hybrid_property[_TE]:
            return self._set(fget=fget)

        def setter(self, fset: _HybridSetterType[_TE]) -> hybrid_property[_TE]:
            return self._set(fset=fset)

        def deleter(self, fdel: _HybridDeleterType[_TE]) -> hybrid_property[_TE]:
            return self._set(fdel=fdel)

        def expression(self, expr: _HybridExprCallableType[_TE]) -> hybrid_property[_TE]:
            return self._set(expr=expr)

        def comparator(self, comparator: _HybridComparatorCallableType[_TE]) -> hybrid_property[_TE]:
            return self._set(custom_comparator=comparator)

        def update_expression(self, meth: _HybridUpdaterType[_TE]) -> hybrid_property[_TE]:
            return self._set(update_expr=meth)

    @property
    def inplace(self) -> _InPlace[_T]:
        """Return the inplace mutator for this :class:`.hybrid_property`.

        This is to allow in-place mutation of the hybrid, allowing the first
        hybrid method of a certain name to be re-used in order to add
        more methods without having to name those methods the same, e.g.::

            class Interval(Base):
                # ...

                @hybrid_property
                def radius(self) -> float:
                    return abs(self.length) / 2

                @radius.inplace.setter
                def _radius_setter(self, value: float) -> None:
                    self.length = value * 2

                @radius.inplace.expression
                def _radius_expression(cls) -> ColumnElement[float]:
                    return type_coerce(func.abs(cls.length) / 2, Float)

        .. versionadded:: 2.0.4

        .. seealso::

            :ref:`hybrid_pep484_naming`

        """
        return hybrid_property._InPlace(self)

    def getter(self, fget: _HybridGetterType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a getter method.

        .. versionadded:: 1.2

        """
        return self._copy(fget=fget)

    def setter(self, fset: _HybridSetterType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a setter method."""
        return self._copy(fset=fset)

    def deleter(self, fdel: _HybridDeleterType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a deletion method."""
        return self._copy(fdel=fdel)

    def expression(self, expr: _HybridExprCallableType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a SQL-expression
        producing method.

        When a hybrid is invoked at the class level, the SQL expression given
        here is wrapped inside of a specialized :class:`.QueryableAttribute`,
        which is the same kind of object used by the ORM to represent other
        mapped attributes.   The reason for this is so that other class-level
        attributes such as docstrings and a reference to the hybrid itself may
        be maintained within the structure that's returned, without any
        modifications to the original SQL expression passed in.

        .. note::

           When referring to a hybrid property  from an owning class (e.g.
           ``SomeClass.some_hybrid``), an instance of
           :class:`.QueryableAttribute` is returned, representing the
           expression or comparator object as well as this  hybrid object.
           However, that object itself has accessors called ``expression`` and
           ``comparator``; so when attempting to override these decorators on a
           subclass, it may be necessary to qualify it using the
           :attr:`.hybrid_property.overrides` modifier first.  See that
           modifier for details.

        .. seealso::

            :ref:`hybrid_distinct_expression`

        """
        return self._copy(expr=expr)

    def comparator(self, comparator: _HybridComparatorCallableType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a custom
        comparator producing method.

        The return value of the decorated method should be an instance of
        :class:`~.hybrid.Comparator`.

        .. note::  The :meth:`.hybrid_property.comparator` decorator
           **replaces** the use of the :meth:`.hybrid_property.expression`
           decorator.  They cannot be used together.

        When a hybrid is invoked at the class level, the
        :class:`~.hybrid.Comparator` object given here is wrapped inside of a
        specialized :class:`.QueryableAttribute`, which is the same kind of
        object used by the ORM to represent other mapped attributes.   The
        reason for this is so that other class-level attributes such as
        docstrings and a reference to the hybrid itself may be maintained
        within the structure that's returned, without any modifications to the
        original comparator object passed in.

        .. note::

           When referring to a hybrid property  from an owning class (e.g.
           ``SomeClass.some_hybrid``), an instance of
           :class:`.QueryableAttribute` is returned, representing the
           expression or comparator object as this  hybrid object.  However,
           that object itself has accessors called ``expression`` and
           ``comparator``; so when attempting to override these decorators on a
           subclass, it may be necessary to qualify it using the
           :attr:`.hybrid_property.overrides` modifier first.  See that
           modifier for details.

        """
        return self._copy(custom_comparator=comparator)

    def update_expression(self, meth: _HybridUpdaterType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines an UPDATE tuple
        producing method.

        The method accepts a single value, which is the value to be
        rendered into the SET clause of an UPDATE statement.  The method
        should then process this value into individual column expressions
        that fit into the ultimate SET clause, and return them as a
        sequence of 2-tuples.  Each tuple
        contains a column expression as the key and a value to be rendered.

        E.g.::

            class Person(Base):
                # ...

                first_name = Column(String)
                last_name = Column(String)

                @hybrid_property
                def fullname(self):
                    return first_name + " " + last_name

                @fullname.update_expression
                def fullname(cls, value):
                    fname, lname = value.split(" ", 1)
                    return [
                        (cls.first_name, fname),
                        (cls.last_name, lname)
                    ]

        .. versionadded:: 1.2

        """
        return self._copy(update_expr=meth)

    @util.memoized_property
    def _expr_comparator(self) -> Callable[[Any], _HybridClassLevelAccessor[_T]]:
        if self.custom_comparator is not None:
            return self._get_comparator(self.custom_comparator)
        elif self.expr is not None:
            return self._get_expr(self.expr)
        else:
            return self._get_expr(cast(_HybridExprCallableType[_T], self.fget))

    def _get_expr(self, expr: _HybridExprCallableType[_T]) -> Callable[[Any], _HybridClassLevelAccessor[_T]]:

        def _expr(cls: Any) -> ExprComparator[_T]:
            return ExprComparator(cls, expr(cls), self)
        util.update_wrapper(_expr, expr)
        return self._get_comparator(_expr)

    def _get_comparator(self, comparator: Any) -> Callable[[Any], _HybridClassLevelAccessor[_T]]:
        proxy_attr = attributes.create_proxied_attribute(self)

        def expr_comparator(owner: Type[object]) -> _HybridClassLevelAccessor[_T]:
            for lookup in owner.__mro__:
                if self.__name__ in lookup.__dict__:
                    if lookup.__dict__[self.__name__] is self:
                        name = self.__name__
                        break
            else:
                name = attributes._UNKNOWN_ATTR_KEY
            return cast('_HybridClassLevelAccessor[_T]', proxy_attr(owner, name, self, comparator(owner), doc=comparator.__doc__ or self.__doc__))
        return expr_comparator
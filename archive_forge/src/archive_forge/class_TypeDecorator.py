from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
class TypeDecorator(SchemaEventTarget, ExternalType, TypeEngine[_T]):
    """Allows the creation of types which add additional functionality
    to an existing type.

    This method is preferred to direct subclassing of SQLAlchemy's
    built-in types as it ensures that all required functionality of
    the underlying type is kept in place.

    Typical usage::

      import sqlalchemy.types as types

      class MyType(types.TypeDecorator):
          '''Prefixes Unicode values with "PREFIX:" on the way in and
          strips it off on the way out.
          '''

          impl = types.Unicode

          cache_ok = True

          def process_bind_param(self, value, dialect):
              return "PREFIX:" + value

          def process_result_value(self, value, dialect):
              return value[7:]

          def copy(self, **kw):
              return MyType(self.impl.length)

    The class-level ``impl`` attribute is required, and can reference any
    :class:`.TypeEngine` class.  Alternatively, the :meth:`load_dialect_impl`
    method can be used to provide different type classes based on the dialect
    given; in this case, the ``impl`` variable can reference
    ``TypeEngine`` as a placeholder.

    The :attr:`.TypeDecorator.cache_ok` class-level flag indicates if this
    custom :class:`.TypeDecorator` is safe to be used as part of a cache key.
    This flag defaults to ``None`` which will initially generate a warning
    when the SQL compiler attempts to generate a cache key for a statement
    that uses this type.  If the :class:`.TypeDecorator` is not guaranteed
    to produce the same bind/result behavior and SQL generation
    every time, this flag should be set to ``False``; otherwise if the
    class produces the same behavior each time, it may be set to ``True``.
    See :attr:`.TypeDecorator.cache_ok` for further notes on how this works.

    Types that receive a Python type that isn't similar to the ultimate type
    used may want to define the :meth:`TypeDecorator.coerce_compared_value`
    method. This is used to give the expression system a hint when coercing
    Python objects into bind parameters within expressions. Consider this
    expression::

        mytable.c.somecol + datetime.date(2009, 5, 15)

    Above, if "somecol" is an ``Integer`` variant, it makes sense that
    we're doing date arithmetic, where above is usually interpreted
    by databases as adding a number of days to the given date.
    The expression system does the right thing by not attempting to
    coerce the "date()" value into an integer-oriented bind parameter.

    However, in the case of ``TypeDecorator``, we are usually changing an
    incoming Python type to something new - ``TypeDecorator`` by default will
    "coerce" the non-typed side to be the same type as itself. Such as below,
    we define an "epoch" type that stores a date value as an integer::

        class MyEpochType(types.TypeDecorator):
            impl = types.Integer

            cache_ok = True

            epoch = datetime.date(1970, 1, 1)

            def process_bind_param(self, value, dialect):
                return (value - self.epoch).days

            def process_result_value(self, value, dialect):
                return self.epoch + timedelta(days=value)

    Our expression of ``somecol + date`` with the above type will coerce the
    "date" on the right side to also be treated as ``MyEpochType``.

    This behavior can be overridden via the
    :meth:`~TypeDecorator.coerce_compared_value` method, which returns a type
    that should be used for the value of the expression. Below we set it such
    that an integer value will be treated as an ``Integer``, and any other
    value is assumed to be a date and will be treated as a ``MyEpochType``::

        def coerce_compared_value(self, op, value):
            if isinstance(value, int):
                return Integer()
            else:
                return self

    .. warning::

       Note that the **behavior of coerce_compared_value is not inherited
       by default from that of the base type**.
       If the :class:`.TypeDecorator` is augmenting a
       type that requires special logic for certain types of operators,
       this method **must** be overridden.  A key example is when decorating
       the :class:`_postgresql.JSON` and :class:`_postgresql.JSONB` types;
       the default rules of :meth:`.TypeEngine.coerce_compared_value` should
       be used in order to deal with operators like index operations::

            from sqlalchemy import JSON
            from sqlalchemy import TypeDecorator

            class MyJsonType(TypeDecorator):
                impl = JSON

                cache_ok = True

                def coerce_compared_value(self, op, value):
                    return self.impl.coerce_compared_value(op, value)

       Without the above step, index operations such as ``mycol['foo']``
       will cause the index value ``'foo'`` to be JSON encoded.

       Similarly, when working with the :class:`.ARRAY` datatype, the
       type coercion for index operations (e.g. ``mycol[5]``) is also
       handled by :meth:`.TypeDecorator.coerce_compared_value`, where
       again a simple override is sufficient unless special rules are needed
       for particular operators::

            from sqlalchemy import ARRAY
            from sqlalchemy import TypeDecorator

            class MyArrayType(TypeDecorator):
                impl = ARRAY

                cache_ok = True

                def coerce_compared_value(self, op, value):
                    return self.impl.coerce_compared_value(op, value)


    """
    __visit_name__ = 'type_decorator'
    _is_type_decorator = True
    impl: Union[TypeEngine[Any], Type[TypeEngine[Any]]]

    @util.memoized_property
    def impl_instance(self) -> TypeEngine[Any]:
        return self.impl

    def __init__(self, *args: Any, **kwargs: Any):
        """Construct a :class:`.TypeDecorator`.

        Arguments sent here are passed to the constructor
        of the class assigned to the ``impl`` class level attribute,
        assuming the ``impl`` is a callable, and the resulting
        object is assigned to the ``self.impl`` instance attribute
        (thus overriding the class attribute of the same name).

        If the class level ``impl`` is not a callable (the unusual case),
        it will be assigned to the same instance attribute 'as-is',
        ignoring those arguments passed to the constructor.

        Subclasses can override this to customize the generation
        of ``self.impl`` entirely.

        """
        if not hasattr(self.__class__, 'impl'):
            raise AssertionError("TypeDecorator implementations require a class-level variable 'impl' which refers to the class of type being decorated")
        self.impl = to_instance(self.__class__.impl, *args, **kwargs)
    coerce_to_is_types: Sequence[Type[Any]] = (type(None),)
    'Specify those Python types which should be coerced at the expression\n    level to "IS <constant>" when compared using ``==`` (and same for\n    ``IS NOT`` in conjunction with ``!=``).\n\n    For most SQLAlchemy types, this includes ``NoneType``, as well as\n    ``bool``.\n\n    :class:`.TypeDecorator` modifies this list to only include ``NoneType``,\n    as typedecorator implementations that deal with boolean types are common.\n\n    Custom :class:`.TypeDecorator` classes can override this attribute to\n    return an empty tuple, in which case no values will be coerced to\n    constants.\n\n    '

    class Comparator(TypeEngine.Comparator[_CT]):
        """A :class:`.TypeEngine.Comparator` that is specific to
        :class:`.TypeDecorator`.

        User-defined :class:`.TypeDecorator` classes should not typically
        need to modify this.


        """
        __slots__ = ()

        def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            if TYPE_CHECKING:
                assert isinstance(self.expr.type, TypeDecorator)
            kwargs['_python_is_types'] = self.expr.type.coerce_to_is_types
            return super().operate(op, *other, **kwargs)

        def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[_CT]:
            if TYPE_CHECKING:
                assert isinstance(self.expr.type, TypeDecorator)
            kwargs['_python_is_types'] = self.expr.type.coerce_to_is_types
            return super().reverse_operate(op, other, **kwargs)

    @property
    def comparator_factory(self) -> _ComparatorFactory[Any]:
        if TypeDecorator.Comparator in self.impl.comparator_factory.__mro__:
            return self.impl.comparator_factory
        else:
            return type('TDComparator', (TypeDecorator.Comparator, self.impl.comparator_factory), {})

    def _gen_dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        if dialect.name in self._variant_mapping:
            adapted = dialect.type_descriptor(self._variant_mapping[dialect.name])
        else:
            adapted = dialect.type_descriptor(self)
        if adapted is not self:
            return adapted
        typedesc = self.load_dialect_impl(dialect).dialect_impl(dialect)
        tt = self.copy()
        if not isinstance(tt, self.__class__):
            raise AssertionError('Type object %s does not properly implement the copy() method, it must return an object of type %s' % (self, self.__class__))
        tt.impl = tt.impl_instance = typedesc
        return tt

    @util.ro_non_memoized_property
    def _type_affinity(self) -> Optional[Type[TypeEngine[Any]]]:
        return self.impl_instance._type_affinity

    def _set_parent(self, parent: SchemaEventTarget, outer: bool=False, **kw: Any) -> None:
        """Support SchemaEventTarget"""
        super()._set_parent(parent)
        if not outer and isinstance(self.impl_instance, SchemaEventTarget):
            self.impl_instance._set_parent(parent, outer=False, **kw)

    def _set_parent_with_dispatch(self, parent: SchemaEventTarget, **kw: Any) -> None:
        """Support SchemaEventTarget"""
        super()._set_parent_with_dispatch(parent, outer=True, **kw)
        if isinstance(self.impl_instance, SchemaEventTarget):
            self.impl_instance._set_parent_with_dispatch(parent)

    def type_engine(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return a dialect-specific :class:`.TypeEngine` instance
        for this :class:`.TypeDecorator`.

        In most cases this returns a dialect-adapted form of
        the :class:`.TypeEngine` type represented by ``self.impl``.
        Makes usage of :meth:`dialect_impl`.
        Behavior can be customized here by overriding
        :meth:`load_dialect_impl`.

        """
        adapted = dialect.type_descriptor(self)
        if not isinstance(adapted, type(self)):
            return adapted
        else:
            return self.load_dialect_impl(dialect)

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return a :class:`.TypeEngine` object corresponding to a dialect.

        This is an end-user override hook that can be used to provide
        differing types depending on the given dialect.  It is used
        by the :class:`.TypeDecorator` implementation of :meth:`type_engine`
        to help determine what type should ultimately be returned
        for a given :class:`.TypeDecorator`.

        By default returns ``self.impl``.

        """
        return self.impl_instance

    def _unwrapped_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return the 'unwrapped' dialect impl for this type.

        This is used by the :meth:`.DefaultDialect.set_input_sizes`
        method.

        """
        typ = self.dialect_impl(dialect)
        if isinstance(typ, self.__class__):
            return typ.load_dialect_impl(dialect).dialect_impl(dialect)
        else:
            return typ

    def __getattr__(self, key: str) -> Any:
        """Proxy all other undefined accessors to the underlying
        implementation."""
        return getattr(self.impl_instance, key)

    def process_literal_param(self, value: Optional[_T], dialect: Dialect) -> str:
        """Receive a literal parameter value to be rendered inline within
        a statement.

        .. note::

            This method is called during the **SQL compilation** phase of a
            statement, when rendering a SQL string. Unlike other SQL
            compilation methods, it is passed a specific Python value to be
            rendered as a string. However it should not be confused with the
            :meth:`_types.TypeDecorator.process_bind_param` method, which is
            the more typical method that processes the actual value passed to a
            particular parameter at statement execution time.

        Custom subclasses of :class:`_types.TypeDecorator` should override
        this method to provide custom behaviors for incoming data values
        that are in the special case of being rendered as literals.

        The returned string will be rendered into the output string.

        """
        raise NotImplementedError()

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:
        """Receive a bound parameter value to be converted.

        Custom subclasses of :class:`_types.TypeDecorator` should override
        this method to provide custom behaviors for incoming data values.
        This method is called at **statement execution time** and is passed
        the literal Python data value which is to be associated with a bound
        parameter in the statement.

        The operation could be anything desired to perform custom
        behavior, such as transforming or serializing data.
        This could also be used as a hook for validating logic.

        :param value: Data to operate upon, of any type expected by
         this method in the subclass.  Can be ``None``.
        :param dialect: the :class:`.Dialect` in use.

        .. seealso::

            :ref:`types_typedecorator`

            :meth:`_types.TypeDecorator.process_result_value`

        """
        raise NotImplementedError()

    def process_result_value(self, value: Optional[Any], dialect: Dialect) -> Optional[_T]:
        """Receive a result-row column value to be converted.

        Custom subclasses of :class:`_types.TypeDecorator` should override
        this method to provide custom behaviors for data values
        being received in result rows coming from the database.
        This method is called at **result fetching time** and is passed
        the literal Python data value that's extracted from a database result
        row.

        The operation could be anything desired to perform custom
        behavior, such as transforming or deserializing data.

        :param value: Data to operate upon, of any type expected by
         this method in the subclass.  Can be ``None``.
        :param dialect: the :class:`.Dialect` in use.

        .. seealso::

            :ref:`types_typedecorator`

            :meth:`_types.TypeDecorator.process_bind_param`


        """
        raise NotImplementedError()

    @util.memoized_property
    def _has_bind_processor(self) -> bool:
        """memoized boolean, check if process_bind_param is implemented.

        Allows the base process_bind_param to raise
        NotImplementedError without needing to test an expensive
        exception throw.

        """
        return util.method_is_overridden(self, TypeDecorator.process_bind_param)

    @util.memoized_property
    def _has_literal_processor(self) -> bool:
        """memoized boolean, check if process_literal_param is implemented."""
        return util.method_is_overridden(self, TypeDecorator.process_literal_param)

    def literal_processor(self, dialect: Dialect) -> Optional[_LiteralProcessorType[_T]]:
        """Provide a literal processing function for the given
        :class:`.Dialect`.

        This is the method that fulfills the :class:`.TypeEngine`
        contract for literal value conversion which normally occurs via
        the :meth:`_types.TypeEngine.literal_processor` method.

        .. note::

            User-defined subclasses of :class:`_types.TypeDecorator` should
            **not** implement this method, and should instead implement
            :meth:`_types.TypeDecorator.process_literal_param` so that the
            "inner" processing provided by the implementing type is maintained.

        """
        if self._has_literal_processor:
            process_literal_param = self.process_literal_param
            process_bind_param = None
        elif self._has_bind_processor:
            process_literal_param = None
            process_bind_param = self.process_bind_param
        else:
            process_literal_param = None
            process_bind_param = None
        if process_literal_param is not None:
            impl_processor = self.impl_instance.literal_processor(dialect)
            if impl_processor:
                fixed_impl_processor = impl_processor
                fixed_process_literal_param = process_literal_param

                def process(value: Any) -> str:
                    return fixed_impl_processor(fixed_process_literal_param(value, dialect))
            else:
                fixed_process_literal_param = process_literal_param

                def process(value: Any) -> str:
                    return fixed_process_literal_param(value, dialect)
            return process
        elif process_bind_param is not None:
            impl_processor = self.impl_instance.literal_processor(dialect)
            if not impl_processor:
                return None
            else:
                fixed_impl_processor = impl_processor
                fixed_process_bind_param = process_bind_param

                def process(value: Any) -> str:
                    return fixed_impl_processor(fixed_process_bind_param(value, dialect))
                return process
        else:
            return self.impl_instance.literal_processor(dialect)

    def bind_processor(self, dialect: Dialect) -> Optional[_BindProcessorType[_T]]:
        """Provide a bound value processing function for the
        given :class:`.Dialect`.

        This is the method that fulfills the :class:`.TypeEngine`
        contract for bound value conversion which normally occurs via
        the :meth:`_types.TypeEngine.bind_processor` method.

        .. note::

            User-defined subclasses of :class:`_types.TypeDecorator` should
            **not** implement this method, and should instead implement
            :meth:`_types.TypeDecorator.process_bind_param` so that the "inner"
            processing provided by the implementing type is maintained.

        :param dialect: Dialect instance in use.

        """
        if self._has_bind_processor:
            process_param = self.process_bind_param
            impl_processor = self.impl_instance.bind_processor(dialect)
            if impl_processor:
                fixed_impl_processor = impl_processor
                fixed_process_param = process_param

                def process(value: Optional[_T]) -> Any:
                    return fixed_impl_processor(fixed_process_param(value, dialect))
            else:
                fixed_process_param = process_param

                def process(value: Optional[_T]) -> Any:
                    return fixed_process_param(value, dialect)
            return process
        else:
            return self.impl_instance.bind_processor(dialect)

    @util.memoized_property
    def _has_result_processor(self) -> bool:
        """memoized boolean, check if process_result_value is implemented.

        Allows the base process_result_value to raise
        NotImplementedError without needing to test an expensive
        exception throw.

        """
        return util.method_is_overridden(self, TypeDecorator.process_result_value)

    def result_processor(self, dialect: Dialect, coltype: Any) -> Optional[_ResultProcessorType[_T]]:
        """Provide a result value processing function for the given
        :class:`.Dialect`.

        This is the method that fulfills the :class:`.TypeEngine`
        contract for bound value conversion which normally occurs via
        the :meth:`_types.TypeEngine.result_processor` method.

        .. note::

            User-defined subclasses of :class:`_types.TypeDecorator` should
            **not** implement this method, and should instead implement
            :meth:`_types.TypeDecorator.process_result_value` so that the
            "inner" processing provided by the implementing type is maintained.

        :param dialect: Dialect instance in use.
        :param coltype: A SQLAlchemy data type

        """
        if self._has_result_processor:
            process_value = self.process_result_value
            impl_processor = self.impl_instance.result_processor(dialect, coltype)
            if impl_processor:
                fixed_process_value = process_value
                fixed_impl_processor = impl_processor

                def process(value: Any) -> Optional[_T]:
                    return fixed_process_value(fixed_impl_processor(value), dialect)
            else:
                fixed_process_value = process_value

                def process(value: Any) -> Optional[_T]:
                    return fixed_process_value(value, dialect)
            return process
        else:
            return self.impl_instance.result_processor(dialect, coltype)

    @util.memoized_property
    def _has_bind_expression(self) -> bool:
        return util.method_is_overridden(self, TypeDecorator.bind_expression) or self.impl_instance._has_bind_expression

    def bind_expression(self, bindparam: BindParameter[_T]) -> Optional[ColumnElement[_T]]:
        """Given a bind value (i.e. a :class:`.BindParameter` instance),
        return a SQL expression which will typically wrap the given parameter.

        .. note::

            This method is called during the **SQL compilation** phase of a
            statement, when rendering a SQL string. It is **not** necessarily
            called against specific values, and should not be confused with the
            :meth:`_types.TypeDecorator.process_bind_param` method, which is
            the more typical method that processes the actual value passed to a
            particular parameter at statement execution time.

        Subclasses of :class:`_types.TypeDecorator` can override this method
        to provide custom bind expression behavior for the type.  This
        implementation will **replace** that of the underlying implementation
        type.

        """
        return self.impl_instance.bind_expression(bindparam)

    @util.memoized_property
    def _has_column_expression(self) -> bool:
        """memoized boolean, check if column_expression is implemented.

        Allows the method to be skipped for the vast majority of expression
        types that don't use this feature.

        """
        return util.method_is_overridden(self, TypeDecorator.column_expression) or self.impl_instance._has_column_expression

    def column_expression(self, column: ColumnElement[_T]) -> Optional[ColumnElement[_T]]:
        """Given a SELECT column expression, return a wrapping SQL expression.

        .. note::

            This method is called during the **SQL compilation** phase of a
            statement, when rendering a SQL string. It is **not** called
            against specific values, and should not be confused with the
            :meth:`_types.TypeDecorator.process_result_value` method, which is
            the more typical method that processes the actual value returned
            in a result row subsequent to statement execution time.

        Subclasses of :class:`_types.TypeDecorator` can override this method
        to provide custom column expression behavior for the type.  This
        implementation will **replace** that of the underlying implementation
        type.

        See the description of :meth:`_types.TypeEngine.column_expression`
        for a complete description of the method's use.

        """
        return self.impl_instance.column_expression(column)

    def coerce_compared_value(self, op: Optional[OperatorType], value: Any) -> Any:
        """Suggest a type for a 'coerced' Python value in an expression.

        By default, returns self.   This method is called by
        the expression system when an object using this type is
        on the left or right side of an expression against a plain Python
        object which does not yet have a SQLAlchemy type assigned::

            expr = table.c.somecolumn + 35

        Where above, if ``somecolumn`` uses this type, this method will
        be called with the value ``operator.add``
        and ``35``.  The return value is whatever SQLAlchemy type should
        be used for ``35`` for this particular operation.

        """
        return self

    def copy(self, **kw: Any) -> Self:
        """Produce a copy of this :class:`.TypeDecorator` instance.

        This is a shallow copy and is provided to fulfill part of
        the :class:`.TypeEngine` contract.  It usually does not
        need to be overridden unless the user-defined :class:`.TypeDecorator`
        has local state that should be deep-copied.

        """
        instance = self.__class__.__new__(self.__class__)
        instance.__dict__.update(self.__dict__)
        return instance

    def get_dbapi_type(self, dbapi: ModuleType) -> Optional[Any]:
        """Return the DBAPI type object represented by this
        :class:`.TypeDecorator`.

        By default this calls upon :meth:`.TypeEngine.get_dbapi_type` of the
        underlying "impl".
        """
        return self.impl_instance.get_dbapi_type(dbapi)

    def compare_values(self, x: Any, y: Any) -> bool:
        """Given two values, compare them for equality.

        By default this calls upon :meth:`.TypeEngine.compare_values`
        of the underlying "impl", which in turn usually
        uses the Python equals operator ``==``.

        This function is used by the ORM to compare
        an original-loaded value with an intercepted
        "changed" value, to determine if a net change
        has occurred.

        """
        return self.impl_instance.compare_values(x, y)

    @property
    def sort_key_function(self) -> Optional[Callable[[Any], Any]]:
        return self.impl_instance.sort_key_function

    def __repr__(self) -> str:
        return util.generic_repr(self, to_inspect=self.impl_instance)
from __future__ import annotations
import collections
import contextlib
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from typing import Union
from ... import exc as sa_exc
from ...engine import Connection
from ...engine import Engine
from ...orm import exc as orm_exc
from ...orm import relationships
from ...orm.base import _mapper_or_none
from ...orm.clsregistry import _resolver
from ...orm.decl_base import _DeferredMapperConfig
from ...orm.util import polymorphic_union
from ...schema import Table
from ...util import OrderedDict
class DeferredReflection:
    """A helper class for construction of mappings based on
    a deferred reflection step.

    Normally, declarative can be used with reflection by
    setting a :class:`_schema.Table` object using autoload_with=engine
    as the ``__table__`` attribute on a declarative class.
    The caveat is that the :class:`_schema.Table` must be fully
    reflected, or at the very least have a primary key column,
    at the point at which a normal declarative mapping is
    constructed, meaning the :class:`_engine.Engine` must be available
    at class declaration time.

    The :class:`.DeferredReflection` mixin moves the construction
    of mappers to be at a later point, after a specific
    method is called which first reflects all :class:`_schema.Table`
    objects created so far.   Classes can define it as such::

        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.ext.declarative import DeferredReflection
        Base = declarative_base()

        class MyClass(DeferredReflection, Base):
            __tablename__ = 'mytable'

    Above, ``MyClass`` is not yet mapped.   After a series of
    classes have been defined in the above fashion, all tables
    can be reflected and mappings created using
    :meth:`.prepare`::

        engine = create_engine("someengine://...")
        DeferredReflection.prepare(engine)

    The :class:`.DeferredReflection` mixin can be applied to individual
    classes, used as the base for the declarative base itself,
    or used in a custom abstract class.   Using an abstract base
    allows that only a subset of classes to be prepared for a
    particular prepare step, which is necessary for applications
    that use more than one engine.  For example, if an application
    has two engines, you might use two bases, and prepare each
    separately, e.g.::

        class ReflectedOne(DeferredReflection, Base):
            __abstract__ = True

        class ReflectedTwo(DeferredReflection, Base):
            __abstract__ = True

        class MyClass(ReflectedOne):
            __tablename__ = 'mytable'

        class MyOtherClass(ReflectedOne):
            __tablename__ = 'myothertable'

        class YetAnotherClass(ReflectedTwo):
            __tablename__ = 'yetanothertable'

        # ... etc.

    Above, the class hierarchies for ``ReflectedOne`` and
    ``ReflectedTwo`` can be configured separately::

        ReflectedOne.prepare(engine_one)
        ReflectedTwo.prepare(engine_two)

    .. seealso::

        :ref:`orm_declarative_reflected_deferred_reflection` - in the
        :ref:`orm_declarative_table_config_toplevel` section.

    """

    @classmethod
    def prepare(cls, bind: Union[Engine, Connection], **reflect_kw: Any) -> None:
        """Reflect all :class:`_schema.Table` objects for all current
        :class:`.DeferredReflection` subclasses

        :param bind: :class:`_engine.Engine` or :class:`_engine.Connection`
         instance

         ..versionchanged:: 2.0.16 a :class:`_engine.Connection` is also
         accepted.

        :param \\**reflect_kw: additional keyword arguments passed to
         :meth:`_schema.MetaData.reflect`, such as
         :paramref:`_schema.MetaData.reflect.views`.

         .. versionadded:: 2.0.16

        """
        to_map = _DeferredMapperConfig.classes_for_base(cls)
        metadata_to_table = collections.defaultdict(set)
        for thingy in to_map:
            if thingy.local_table is not None:
                metadata_to_table[thingy.local_table.metadata, thingy.local_table.schema].add(thingy.local_table.name)
        if isinstance(bind, Connection):
            conn = bind
            ctx = contextlib.nullcontext(enter_result=conn)
        elif isinstance(bind, Engine):
            ctx = bind.connect()
        else:
            raise sa_exc.ArgumentError(f'Expected Engine or Connection, got {bind!r}')
        with ctx as conn:
            for (metadata, schema), table_names in metadata_to_table.items():
                metadata.reflect(conn, only=table_names, schema=schema, extend_existing=True, autoload_replace=False, **reflect_kw)
            metadata_to_table.clear()
            for thingy in to_map:
                thingy.map()
                mapper = thingy.cls.__mapper__
                metadata = mapper.class_.metadata
                for rel in mapper._props.values():
                    if isinstance(rel, relationships.RelationshipProperty) and rel._init_args.secondary._is_populated():
                        secondary_arg = rel._init_args.secondary
                        if isinstance(secondary_arg.argument, Table):
                            secondary_table = secondary_arg.argument
                            metadata_to_table[secondary_table.metadata, secondary_table.schema].add(secondary_table.name)
                        elif isinstance(secondary_arg.argument, str):
                            _, resolve_arg = _resolver(rel.parent.class_, rel)
                            resolver = resolve_arg(secondary_arg.argument, True)
                            metadata_to_table[metadata, thingy.local_table.schema].add(secondary_arg.argument)
                            resolver._resolvers += (cls._sa_deferred_table_resolver(metadata),)
                            secondary_arg.argument = resolver()
            for (metadata, schema), table_names in metadata_to_table.items():
                metadata.reflect(conn, only=table_names, schema=schema, extend_existing=True, autoload_replace=False)

    @classmethod
    def _sa_deferred_table_resolver(cls, metadata: MetaData) -> Callable[[str], Table]:

        def _resolve(key: str) -> Table:
            return Table(key, metadata)
        return _resolve
    _sa_decl_prepare = True

    @classmethod
    def _sa_raise_deferred_config(cls):
        raise orm_exc.UnmappedClassError(cls, msg='Class %s is a subclass of DeferredReflection.  Mappings are not produced until the .prepare() method is called on the class hierarchy.' % orm_exc._safe_cls_name(cls))
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
class AbstractConcreteBase(ConcreteBase):
    """A helper class for 'concrete' declarative mappings.

    :class:`.AbstractConcreteBase` will use the :func:`.polymorphic_union`
    function automatically, against all tables mapped as a subclass
    to this class.   The function is called via the
    ``__declare_first__()`` function, which is essentially
    a hook for the :meth:`.before_configured` event.

    :class:`.AbstractConcreteBase` applies :class:`_orm.Mapper` for its
    immediately inheriting class, as would occur for any other
    declarative mapped class. However, the :class:`_orm.Mapper` is not
    mapped to any particular :class:`.Table` object.  Instead, it's
    mapped directly to the "polymorphic" selectable produced by
    :func:`.polymorphic_union`, and performs no persistence operations on its
    own.  Compare to :class:`.ConcreteBase`, which maps its
    immediately inheriting class to an actual
    :class:`.Table` that stores rows directly.

    .. note::

        The :class:`.AbstractConcreteBase` delays the mapper creation of the
        base class until all the subclasses have been defined,
        as it needs to create a mapping against a selectable that will include
        all subclass tables.  In order to achieve this, it waits for the
        **mapper configuration event** to occur, at which point it scans
        through all the configured subclasses and sets up a mapping that will
        query against all subclasses at once.

        While this event is normally invoked automatically, in the case of
        :class:`.AbstractConcreteBase`, it may be necessary to invoke it
        explicitly after **all** subclass mappings are defined, if the first
        operation is to be a query against this base class. To do so, once all
        the desired classes have been configured, the
        :meth:`_orm.registry.configure` method on the :class:`_orm.registry`
        in use can be invoked, which is available in relation to a particular
        declarative base class::

            Base.registry.configure()

    Example::

        from sqlalchemy.orm import DeclarativeBase
        from sqlalchemy.ext.declarative import AbstractConcreteBase

        class Base(DeclarativeBase):
            pass

        class Employee(AbstractConcreteBase, Base):
            pass

        class Manager(Employee):
            __tablename__ = 'manager'
            employee_id = Column(Integer, primary_key=True)
            name = Column(String(50))
            manager_data = Column(String(40))

            __mapper_args__ = {
                'polymorphic_identity':'manager',
                'concrete':True
            }

        Base.registry.configure()

    The abstract base class is handled by declarative in a special way;
    at class configuration time, it behaves like a declarative mixin
    or an ``__abstract__`` base class.   Once classes are configured
    and mappings are produced, it then gets mapped itself, but
    after all of its descendants.  This is a very unique system of mapping
    not found in any other SQLAlchemy API feature.

    Using this approach, we can specify columns and properties
    that will take place on mapped subclasses, in the way that
    we normally do as in :ref:`declarative_mixins`::

        from sqlalchemy.ext.declarative import AbstractConcreteBase

        class Company(Base):
            __tablename__ = 'company'
            id = Column(Integer, primary_key=True)

        class Employee(AbstractConcreteBase, Base):
            strict_attrs = True

            employee_id = Column(Integer, primary_key=True)

            @declared_attr
            def company_id(cls):
                return Column(ForeignKey('company.id'))

            @declared_attr
            def company(cls):
                return relationship("Company")

        class Manager(Employee):
            __tablename__ = 'manager'

            name = Column(String(50))
            manager_data = Column(String(40))

            __mapper_args__ = {
                'polymorphic_identity':'manager',
                'concrete':True
            }

        Base.registry.configure()

    When we make use of our mappings however, both ``Manager`` and
    ``Employee`` will have an independently usable ``.company`` attribute::

        session.execute(
            select(Employee).filter(Employee.company.has(id=5))
        )

    :param strict_attrs: when specified on the base class, "strict" attribute
     mode is enabled which attempts to limit ORM mapped attributes on the
     base class to only those that are immediately present, while still
     preserving "polymorphic" loading behavior.

     .. versionadded:: 2.0

    .. seealso::

        :class:`.ConcreteBase`

        :ref:`concrete_inheritance`

        :ref:`abstract_concrete_base`

    """
    __no_table__ = True

    @classmethod
    def __declare_first__(cls):
        cls._sa_decl_prepare_nocascade()

    @classmethod
    def _sa_decl_prepare_nocascade(cls):
        if getattr(cls, '__mapper__', None):
            return
        to_map = _DeferredMapperConfig.config_for_cls(cls)
        mappers = []
        stack = list(cls.__subclasses__())
        while stack:
            klass = stack.pop()
            stack.extend(klass.__subclasses__())
            mn = _mapper_or_none(klass)
            if mn is not None:
                mappers.append(mn)
        discriminator_name = getattr(cls, '_concrete_discriminator_name', None) or 'type'
        pjoin = cls._create_polymorphic_union(mappers, discriminator_name)
        declared_cols = set(to_map.declared_columns)
        declared_col_keys = {c.key for c in declared_cols}
        for k, v in list(to_map.properties.items()):
            if v in declared_cols:
                to_map.properties[k] = pjoin.c[v.key]
                declared_col_keys.remove(v.key)
        to_map.local_table = pjoin
        strict_attrs = cls.__dict__.get('strict_attrs', False)
        m_args = to_map.mapper_args_fn or dict

        def mapper_args():
            args = m_args()
            args['polymorphic_on'] = pjoin.c[discriminator_name]
            args['polymorphic_abstract'] = True
            if strict_attrs:
                args['include_properties'] = set(pjoin.primary_key) | declared_col_keys | {discriminator_name}
                args['with_polymorphic'] = ('*', pjoin)
            return args
        to_map.mapper_args_fn = mapper_args
        to_map.map()
        stack = [cls]
        while stack:
            scls = stack.pop(0)
            stack.extend(scls.__subclasses__())
            sm = _mapper_or_none(scls)
            if sm and sm.concrete and (sm.inherits is None):
                for sup_ in scls.__mro__[1:]:
                    sup_sm = _mapper_or_none(sup_)
                    if sup_sm:
                        sm._set_concrete_base(sup_sm)
                        break

    @classmethod
    def _sa_raise_deferred_config(cls):
        raise orm_exc.UnmappedClassError(cls, msg='Class %s is a subclass of AbstractConcreteBase and has a mapping pending until all subclasses are defined. Call the sqlalchemy.orm.configure_mappers() function after all subclasses have been defined to complete the mapping of this class.' % orm_exc._safe_cls_name(cls))
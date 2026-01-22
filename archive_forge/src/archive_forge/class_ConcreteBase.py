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
class ConcreteBase:
    """A helper class for 'concrete' declarative mappings.

    :class:`.ConcreteBase` will use the :func:`.polymorphic_union`
    function automatically, against all tables mapped as a subclass
    to this class.   The function is called via the
    ``__declare_last__()`` function, which is essentially
    a hook for the :meth:`.after_configured` event.

    :class:`.ConcreteBase` produces a mapped
    table for the class itself.  Compare to :class:`.AbstractConcreteBase`,
    which does not.

    Example::

        from sqlalchemy.ext.declarative import ConcreteBase

        class Employee(ConcreteBase, Base):
            __tablename__ = 'employee'
            employee_id = Column(Integer, primary_key=True)
            name = Column(String(50))
            __mapper_args__ = {
                            'polymorphic_identity':'employee',
                            'concrete':True}

        class Manager(Employee):
            __tablename__ = 'manager'
            employee_id = Column(Integer, primary_key=True)
            name = Column(String(50))
            manager_data = Column(String(40))
            __mapper_args__ = {
                            'polymorphic_identity':'manager',
                            'concrete':True}


    The name of the discriminator column used by :func:`.polymorphic_union`
    defaults to the name ``type``.  To suit the use case of a mapping where an
    actual column in a mapped table is already named ``type``, the
    discriminator name can be configured by setting the
    ``_concrete_discriminator_name`` attribute::

        class Employee(ConcreteBase, Base):
            _concrete_discriminator_name = '_concrete_discriminator'

    .. versionadded:: 1.3.19 Added the ``_concrete_discriminator_name``
       attribute to :class:`_declarative.ConcreteBase` so that the
       virtual discriminator column name can be customized.

    .. versionchanged:: 1.4.2 The ``_concrete_discriminator_name`` attribute
       need only be placed on the basemost class to take correct effect for
       all subclasses.   An explicit error message is now raised if the
       mapped column names conflict with the discriminator name, whereas
       in the 1.3.x series there would be some warnings and then a non-useful
       query would be generated.

    .. seealso::

        :class:`.AbstractConcreteBase`

        :ref:`concrete_inheritance`


    """

    @classmethod
    def _create_polymorphic_union(cls, mappers, discriminator_name):
        return polymorphic_union(OrderedDict(((mp.polymorphic_identity, mp.local_table) for mp in mappers)), discriminator_name, 'pjoin')

    @classmethod
    def __declare_first__(cls):
        m = cls.__mapper__
        if m.with_polymorphic:
            return
        discriminator_name = getattr(cls, '_concrete_discriminator_name', None) or 'type'
        mappers = list(m.self_and_descendants)
        pjoin = cls._create_polymorphic_union(mappers, discriminator_name)
        m._set_with_polymorphic(('*', pjoin))
        m._set_polymorphic_on(pjoin.c[discriminator_name])
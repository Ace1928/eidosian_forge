from .extensions import AbstractConcreteBase
from .extensions import ConcreteBase
from .extensions import DeferredReflection
from ... import util
from ...orm.decl_api import as_declarative as _as_declarative
from ...orm.decl_api import declarative_base as _declarative_base
from ...orm.decl_api import DeclarativeMeta
from ...orm.decl_api import declared_attr
from ...orm.decl_api import has_inherited_table as _has_inherited_table
from ...orm.decl_api import synonym_for as _synonym_for
@util.moved_20('The ``as_declarative()`` function is now available as :func:`sqlalchemy.orm.as_declarative`')
def as_declarative(*arg, **kw):
    return _as_declarative(*arg, **kw)
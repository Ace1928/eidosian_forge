from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
class InspectionAttr:
    """A base class applied to all ORM objects and attributes that are
    related to things that can be returned by the :func:`_sa.inspect` function.

    The attributes defined here allow the usage of simple boolean
    checks to test basic facts about the object returned.

    While the boolean checks here are basically the same as using
    the Python isinstance() function, the flags here can be used without
    the need to import all of these classes, and also such that
    the SQLAlchemy class system can change while leaving the flags
    here intact for forwards-compatibility.

    """
    __slots__: Tuple[str, ...] = ()
    is_selectable = False
    'Return True if this object is an instance of\n    :class:`_expression.Selectable`.'
    is_aliased_class = False
    'True if this object is an instance of :class:`.AliasedClass`.'
    is_instance = False
    'True if this object is an instance of :class:`.InstanceState`.'
    is_mapper = False
    'True if this object is an instance of :class:`_orm.Mapper`.'
    is_bundle = False
    'True if this object is an instance of :class:`.Bundle`.'
    is_property = False
    'True if this object is an instance of :class:`.MapperProperty`.'
    is_attribute = False
    'True if this object is a Python :term:`descriptor`.\n\n    This can refer to one of many types.   Usually a\n    :class:`.QueryableAttribute` which handles attributes events on behalf\n    of a :class:`.MapperProperty`.   But can also be an extension type\n    such as :class:`.AssociationProxy` or :class:`.hybrid_property`.\n    The :attr:`.InspectionAttr.extension_type` will refer to a constant\n    identifying the specific subtype.\n\n    .. seealso::\n\n        :attr:`_orm.Mapper.all_orm_descriptors`\n\n    '
    _is_internal_proxy = False
    'True if this object is an internal proxy object.\n\n    .. versionadded:: 1.2.12\n\n    '
    is_clause_element = False
    'True if this object is an instance of\n    :class:`_expression.ClauseElement`.'
    extension_type: InspectionAttrExtensionType = NotExtension.NOT_EXTENSION
    'The extension type, if any.\n    Defaults to :attr:`.interfaces.NotExtension.NOT_EXTENSION`\n\n    .. seealso::\n\n        :class:`.HybridExtensionType`\n\n        :class:`.AssociationProxyExtensionType`\n\n    '
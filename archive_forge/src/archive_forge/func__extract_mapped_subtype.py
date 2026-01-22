from __future__ import annotations
import enum
import functools
import re
import types
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes  # noqa
from . import exc
from ._typing import _O
from ._typing import insp_is_aliased_class
from ._typing import insp_is_mapper
from ._typing import prop_is_relationship
from .base import _class_to_mapper as _class_to_mapper
from .base import _MappedAnnotationBase
from .base import _never_set as _never_set  # noqa: F401
from .base import _none_set as _none_set  # noqa: F401
from .base import attribute_str as attribute_str  # noqa: F401
from .base import class_mapper as class_mapper
from .base import DynamicMapped
from .base import InspectionAttr as InspectionAttr
from .base import instance_str as instance_str  # noqa: F401
from .base import Mapped
from .base import object_mapper as object_mapper
from .base import object_state as object_state  # noqa: F401
from .base import opt_manager_of_class
from .base import ORMDescriptor
from .base import state_attribute_str as state_attribute_str  # noqa: F401
from .base import state_class_str as state_class_str  # noqa: F401
from .base import state_str as state_str  # noqa: F401
from .base import WriteOnlyMapped
from .interfaces import CriteriaOption
from .interfaces import MapperProperty as MapperProperty
from .interfaces import ORMColumnsClauseRole
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .path_registry import PathRegistry as PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import sql
from .. import util
from ..engine.result import result_tuple
from ..sql import coercions
from ..sql import expression
from ..sql import lambdas
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import is_selectable
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import ColumnCollection
from ..sql.cache_key import HasCacheKey
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import ColumnElement
from ..sql.elements import KeyedColumnElement
from ..sql.selectable import FromClause
from ..util.langhelpers import MemoizedSlots
from ..util.typing import de_stringify_annotation as _de_stringify_annotation
from ..util.typing import (
from ..util.typing import eval_name_only as _eval_name_only
from ..util.typing import is_origin_of_cls
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import typing_get_origin
def _extract_mapped_subtype(raw_annotation: Optional[_AnnotationScanType], cls: type, originating_module: str, key: str, attr_cls: Type[Any], required: bool, is_dataclass_field: bool, expect_mapped: bool=True, raiseerr: bool=True) -> Optional[Tuple[Union[type, str], Optional[type]]]:
    """given an annotation, figure out if it's ``Mapped[something]`` and if
    so, return the ``something`` part.

    Includes error raise scenarios and other options.

    """
    if raw_annotation is None:
        if required:
            raise sa_exc.ArgumentError(f'Python typing annotation is required for attribute "{cls.__name__}.{key}" when primary argument(s) for "{attr_cls.__name__}" construct are None or not present')
        return None
    try:
        annotated = de_stringify_annotation(cls, raw_annotation, originating_module, str_cleanup_fn=_cleanup_mapped_str_annotation)
    except _CleanupError as ce:
        raise sa_exc.ArgumentError(f'Could not interpret annotation {raw_annotation}.  Check that it uses names that are correctly imported at the module level. See chained stack trace for more hints.') from ce
    except NameError as ne:
        if raiseerr and 'Mapped[' in raw_annotation:
            raise sa_exc.ArgumentError(f'Could not interpret annotation {raw_annotation}.  Check that it uses names that are correctly imported at the module level. See chained stack trace for more hints.') from ne
        annotated = raw_annotation
    if is_dataclass_field:
        return (annotated, None)
    else:
        if not hasattr(annotated, '__origin__') or not is_origin_of_cls(annotated, _MappedAnnotationBase):
            if expect_mapped:
                if not raiseerr:
                    return None
                origin = getattr(annotated, '__origin__', None)
                if origin is typing.ClassVar:
                    return None
                elif isinstance(origin, type) and issubclass(origin, ORMDescriptor):
                    return None
                raise sa_exc.ArgumentError(f'''Type annotation for "{cls.__name__}.{key}" can't be correctly interpreted for Annotated Declarative Table form.  ORM annotations should normally make use of the ``Mapped[]`` generic type, or other ORM-compatible generic type, as a container for the actual type, which indicates the intent that the attribute is mapped. Class variables that are not intended to be mapped by the ORM should use ClassVar[].  To allow Annotated Declarative to disregard legacy annotations which don't use Mapped[] to pass, set "__allow_unmapped__ = True" on the class or a superclass this class.''', code='zlpr')
            else:
                return (annotated, None)
        if len(annotated.__args__) != 1:
            raise sa_exc.ArgumentError('Expected sub-type for Mapped[] annotation')
        return (annotated.__args__[0], annotated.__origin__)
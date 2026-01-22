from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
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
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class _anonymous_label(_truncated_label):
    """A unicode subclass used to identify anonymously
    generated names."""
    __slots__ = ()

    @classmethod
    def safe_construct(cls, seed: int, body: str, enclosing_label: Optional[str]=None, sanitize_key: bool=False) -> _anonymous_label:
        body = re.sub('[%\\(\\) \\$]+', '_', body)
        if sanitize_key:
            body = body.strip('_')
        label = '%%(%d %s)s' % (seed, body.replace('%', '%%'))
        if enclosing_label:
            label = '%s%s' % (enclosing_label, label)
        return _anonymous_label(label)

    def __add__(self, other):
        if '%' in other and (not isinstance(other, _anonymous_label)):
            other = str(other).replace('%', '%%')
        else:
            other = str(other)
        return _anonymous_label(quoted_name(str.__add__(self, other), self.quote))

    def __radd__(self, other):
        if '%' in other and (not isinstance(other, _anonymous_label)):
            other = str(other).replace('%', '%%')
        else:
            other = str(other)
        return _anonymous_label(quoted_name(str.__add__(other, self), self.quote))

    def apply_map(self, map_):
        if self.quote is not None:
            return quoted_name(self % map_, self.quote)
        else:
            return self % map_
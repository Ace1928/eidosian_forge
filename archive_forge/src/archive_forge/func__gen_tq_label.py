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
def _gen_tq_label(self, name: str, dedupe_on_key: bool=True) -> Optional[str]:
    """generate table-qualified label

        for a table-bound column this is <tablename>_<columnname>.

        used primarily for LABEL_STYLE_TABLENAME_PLUS_COL
        as well as the .columns collection on a Join object.

        """
    label: str
    t = self.table
    if self.is_literal:
        return None
    elif t is not None and is_named_from_clause(t):
        if has_schema_attr(t) and t.schema:
            label = t.schema.replace('.', '_') + '_' + t.name + '_' + name
        else:
            assert not TYPE_CHECKING or isinstance(t, NamedFromClause)
            label = t.name + '_' + name
        if is_quoted_name(name) and name.quote is not None:
            if is_quoted_name(label):
                label.quote = name.quote
            else:
                label = quoted_name(label, name.quote)
        elif is_quoted_name(t.name) and t.name.quote is not None:
            assert not isinstance(label, quoted_name)
            label = quoted_name(label, t.name.quote)
        if dedupe_on_key:
            if label in t.c:
                _label = label
                counter = 1
                while _label in t.c:
                    _label = label + '_' + str(counter)
                    counter += 1
                label = _label
        return coercions.expect(roles.TruncatedLabelRole, label)
    else:
        return name
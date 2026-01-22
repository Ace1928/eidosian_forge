from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from . import operators
from .cache_key import HasCacheKey
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import util
from ..util.typing import Literal
from ..util.typing import Self
@classmethod
def _as_annotated_instance(cls, element: SupportsWrappingAnnotations, values: _AnnotationDict) -> Annotated:
    try:
        cls = annotated_classes[element.__class__]
    except KeyError:
        cls = _new_annotation_type(element.__class__, cls)
    return cls(element, values)
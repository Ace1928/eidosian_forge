import copy
import re
from collections import Counter as CollectionCounter, defaultdict, deque
from collections.abc import Callable, Hashable as CollectionsHashable, Iterable as CollectionsIterable
from typing import (
from typing_extensions import Annotated, Final
from . import errors as errors_
from .class_validators import Validator, make_generic_validator, prep_validators
from .error_wrappers import ErrorWrapper
from .errors import ConfigError, InvalidDiscriminator, MissingDiscriminator, NoneIsNotAllowedError
from .types import Json, JsonWrapper
from .typing import (
from .utils import (
from .validators import constant_validator, dict_validator, find_validators, validate_json
def _validate_sequence_like(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
    """
        Validate sequence-like containers: lists, tuples, sets and generators
        Note that large if-else blocks are necessary to enable Cython
        optimization, which is why we disable the complexity check above.
        """
    if not sequence_like(v):
        e: errors_.PydanticTypeError
        if self.shape == SHAPE_LIST:
            e = errors_.ListError()
        elif self.shape in (SHAPE_TUPLE, SHAPE_TUPLE_ELLIPSIS):
            e = errors_.TupleError()
        elif self.shape == SHAPE_SET:
            e = errors_.SetError()
        elif self.shape == SHAPE_FROZENSET:
            e = errors_.FrozenSetError()
        else:
            e = errors_.SequenceError()
        return (v, ErrorWrapper(e, loc))
    loc = loc if isinstance(loc, tuple) else (loc,)
    result = []
    errors: List[ErrorList] = []
    for i, v_ in enumerate(v):
        v_loc = (*loc, i)
        r, ee = self._validate_singleton(v_, values, v_loc, cls)
        if ee:
            errors.append(ee)
        else:
            result.append(r)
    if errors:
        return (v, errors)
    converted: Union[List[Any], Set[Any], FrozenSet[Any], Tuple[Any, ...], Iterator[Any], Deque[Any]] = result
    if self.shape == SHAPE_SET:
        converted = set(result)
    elif self.shape == SHAPE_FROZENSET:
        converted = frozenset(result)
    elif self.shape == SHAPE_TUPLE_ELLIPSIS:
        converted = tuple(result)
    elif self.shape == SHAPE_DEQUE:
        converted = deque(result, maxlen=getattr(v, 'maxlen', None))
    elif self.shape == SHAPE_SEQUENCE:
        if isinstance(v, tuple):
            converted = tuple(result)
        elif isinstance(v, set):
            converted = set(result)
        elif isinstance(v, Generator):
            converted = iter(result)
        elif isinstance(v, deque):
            converted = deque(result, maxlen=getattr(v, 'maxlen', None))
    return (converted, None)
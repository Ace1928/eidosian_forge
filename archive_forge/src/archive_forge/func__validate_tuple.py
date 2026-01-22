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
def _validate_tuple(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
    e: Optional[Exception] = None
    if not sequence_like(v):
        e = errors_.TupleError()
    else:
        actual_length, expected_length = (len(v), len(self.sub_fields))
        if actual_length != expected_length:
            e = errors_.TupleLengthError(actual_length=actual_length, expected_length=expected_length)
    if e:
        return (v, ErrorWrapper(e, loc))
    loc = loc if isinstance(loc, tuple) else (loc,)
    result = []
    errors: List[ErrorList] = []
    for i, (v_, field) in enumerate(zip(v, self.sub_fields)):
        v_loc = (*loc, i)
        r, ee = field.validate(v_, values, loc=v_loc, cls=cls)
        if ee:
            errors.append(ee)
        else:
            result.append(r)
    if errors:
        return (v, errors)
    else:
        return (tuple(result), None)
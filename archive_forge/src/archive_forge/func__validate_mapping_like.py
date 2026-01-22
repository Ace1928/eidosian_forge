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
def _validate_mapping_like(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
    try:
        v_iter = dict_validator(v)
    except TypeError as exc:
        return (v, ErrorWrapper(exc, loc))
    loc = loc if isinstance(loc, tuple) else (loc,)
    result, errors = ({}, [])
    for k, v_ in v_iter.items():
        v_loc = (*loc, '__key__')
        key_result, key_errors = self.key_field.validate(k, values, loc=v_loc, cls=cls)
        if key_errors:
            errors.append(key_errors)
            continue
        v_loc = (*loc, k)
        value_result, value_errors = self._validate_singleton(v_, values, v_loc, cls)
        if value_errors:
            errors.append(value_errors)
            continue
        result[key_result] = value_result
    if errors:
        return (v, errors)
    elif self.shape == SHAPE_DICT:
        return (result, None)
    elif self.shape == SHAPE_DEFAULTDICT:
        return (defaultdict(self.type_, result), None)
    elif self.shape == SHAPE_COUNTER:
        return (CollectionCounter(result), None)
    else:
        return (self._get_mapping_value(v, result), None)
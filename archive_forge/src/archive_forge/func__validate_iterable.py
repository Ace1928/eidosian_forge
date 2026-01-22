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
def _validate_iterable(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
    """
        Validate Iterables.

        This intentionally doesn't validate values to allow infinite generators.
        """
    try:
        iterable = iter(v)
    except TypeError:
        return (v, ErrorWrapper(errors_.IterableError(), loc))
    return (iterable, None)
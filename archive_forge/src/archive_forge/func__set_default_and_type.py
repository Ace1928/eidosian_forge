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
def _set_default_and_type(self) -> None:
    """
        Set the default value, infer the type if needed and check if `None` value is valid.
        """
    if self.default_factory is not None:
        if self.type_ is Undefined:
            raise errors_.ConfigError(f'you need to set the type of field {self.name!r} when using `default_factory`')
        return
    default_value = self.get_default()
    if default_value is not None and self.type_ is Undefined:
        self.type_ = default_value.__class__
        self.outer_type_ = self.type_
        self.annotation = self.type_
    if self.type_ is Undefined:
        raise errors_.ConfigError(f'unable to infer type for attribute "{self.name}"')
    if self.required is False and default_value is None:
        self.allow_none = True
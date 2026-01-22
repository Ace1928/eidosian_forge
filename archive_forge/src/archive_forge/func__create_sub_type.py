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
def _create_sub_type(self, type_: Type[Any], name: str, *, for_keys: bool=False) -> 'ModelField':
    if for_keys:
        class_validators = None
    else:
        class_validators = {k: Validator(func=v.func, pre=v.pre, each_item=False, always=v.always, check_fields=v.check_fields, skip_on_failure=v.skip_on_failure) for k, v in self.class_validators.items() if v.each_item}
    field_info, _ = self._get_field_info(name, type_, None, self.model_config)
    return self.__class__(type_=type_, name=name, class_validators=class_validators, model_config=self.model_config, field_info=field_info)
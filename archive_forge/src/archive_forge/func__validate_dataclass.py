import copy
import dataclasses
import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from .class_validators import gather_all_validators
from .config import BaseConfig, ConfigDict, Extra, get_config
from .error_wrappers import ValidationError
from .errors import DataclassTypeError
from .fields import Field, FieldInfo, Required, Undefined
from .main import create_model, validate_model
from .utils import ClassAttribute
def _validate_dataclass(cls: Type['DataclassT'], v: Any) -> 'DataclassT':
    with set_validation(cls, True):
        if isinstance(v, cls):
            v.__pydantic_validate_values__()
            return v
        elif isinstance(v, (list, tuple)):
            return cls(*v)
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise DataclassTypeError(class_name=cls.__name__)
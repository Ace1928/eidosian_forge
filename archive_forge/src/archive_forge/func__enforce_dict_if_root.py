import warnings
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from types import FunctionType, prepare_class, resolve_bases
from typing import (
from typing_extensions import dataclass_transform
from .class_validators import ValidatorGroup, extract_root_validators, extract_validators, inherit_validators
from .config import BaseConfig, Extra, inherit_config, prepare_config
from .error_wrappers import ErrorWrapper, ValidationError
from .errors import ConfigError, DictError, ExtraError, MissingError
from .fields import (
from .json import custom_pydantic_encoder, pydantic_encoder
from .parse import Protocol, load_file, load_str_bytes
from .schema import default_ref_template, model_schema
from .types import PyObject, StrBytes
from .typing import (
from .utils import (
@classmethod
def _enforce_dict_if_root(cls, obj: Any) -> Any:
    if cls.__custom_root_type__ and (not (isinstance(obj, dict) and obj.keys() == {ROOT_KEY}) and (not (isinstance(obj, BaseModel) and obj.__fields__.keys() == {ROOT_KEY})) or cls.__fields__[ROOT_KEY].shape in MAPPING_LIKE_SHAPES):
        return {ROOT_KEY: obj}
    else:
        return obj
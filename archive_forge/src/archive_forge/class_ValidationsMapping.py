from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
class ValidationsMapping:
    """This class just contains mappings from core_schema attribute names to the corresponding
        JSON schema attribute names. While I suspect it is unlikely to be necessary, you can in
        principle override this class in a subclass of GenerateJsonSchema (by inheriting from
        GenerateJsonSchema.ValidationsMapping) to change these mappings.
        """
    numeric = {'multiple_of': 'multipleOf', 'le': 'maximum', 'ge': 'minimum', 'lt': 'exclusiveMaximum', 'gt': 'exclusiveMinimum'}
    bytes = {'min_length': 'minLength', 'max_length': 'maxLength'}
    string = {'min_length': 'minLength', 'max_length': 'maxLength', 'pattern': 'pattern'}
    array = {'min_length': 'minItems', 'max_length': 'maxItems'}
    object = {'min_length': 'minProperties', 'max_length': 'maxProperties'}
    date = {'le': 'maximum', 'ge': 'minimum', 'lt': 'exclusiveMaximum', 'gt': 'exclusiveMinimum'}
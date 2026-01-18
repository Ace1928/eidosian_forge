from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def ts_type_from_python(type_: SCHEMA_TYPE) -> str:
    if type_ is None:
        return 'any'
    elif isinstance(type_, str):
        return {'str': 'string', 'integer': 'number', 'float': 'number', 'date-time': 'string'}.get(type_, type_)
    elif isinstance(type_, tuple):
        return f'Array<{APIOperation.ts_type_from_python(type_[0])}>'
    elif isinstance(type_, type) and issubclass(type_, Enum):
        return ' | '.join([f"'{e.value}'" for e in type_])
    else:
        return str(type_)
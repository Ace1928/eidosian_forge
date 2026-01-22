from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
class APIPropertyBase(BaseModel):
    """Base model for an API property."""
    name: str = Field(alias='name')
    'The name of the property.'
    required: bool = Field(alias='required')
    'Whether the property is required.'
    type: SCHEMA_TYPE = Field(alias='type')
    "The type of the property.\n    \n    Either a primitive type, a component/parameter type,\n    or an array or 'object' (dict) of the above."
    default: Optional[Any] = Field(alias='default', default=None)
    'The default value of the property.'
    description: Optional[str] = Field(alias='description', default=None)
    'The description of the property.'
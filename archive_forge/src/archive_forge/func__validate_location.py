from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def _validate_location(location: APIPropertyLocation, name: str) -> None:
    if location not in SUPPORTED_LOCATIONS:
        raise NotImplementedError(INVALID_LOCATION_TEMPL.format(location=location, name=name))
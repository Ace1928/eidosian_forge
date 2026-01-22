from __future__ import annotations as _annotations
import re
from typing_extensions import Literal, Self
from ._migration import getattr_migration
from .version import version_short
class PydanticInvalidForJsonSchema(PydanticUserError):
    """An error raised during failures to generate a JSON schema for some `CoreSchema`.

    Attributes:
        message: Description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code='invalid-for-json-schema')
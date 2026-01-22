from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import core_schema
from typing_extensions import Literal
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
Resolves reference in the core schema.

        Args:
            maybe_ref_schema: The input core schema that may contains reference.

        Returns:
            Resolved core schema.

        Raises:
            LookupError: If it can't find the definition for reference.
        
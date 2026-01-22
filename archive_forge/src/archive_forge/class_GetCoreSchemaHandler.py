from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Union
from pydantic_core import core_schema
class GetCoreSchemaHandler:
    """Handler to call into the next CoreSchema schema generation function."""

    def __call__(self, __source_type: Any) -> core_schema.CoreSchema:
        """Call the inner handler and get the CoreSchema it returns.
        This will call the next CoreSchema modifying function up until it calls
        into Pydantic's internal schema generation machinery, which will raise a
        `pydantic.errors.PydanticSchemaGenerationError` error if it cannot generate
        a CoreSchema for the given source type.

        Args:
            __source_type: The input type.

        Returns:
            CoreSchema: The `pydantic-core` CoreSchema generated.
        """
        raise NotImplementedError

    def generate_schema(self, __source_type: Any) -> core_schema.CoreSchema:
        """Generate a schema unrelated to the current context.
        Use this function if e.g. you are handling schema generation for a sequence
        and want to generate a schema for its items.
        Otherwise, you may end up doing something like applying a `min_length` constraint
        that was intended for the sequence itself to its items!

        Args:
            __source_type: The input type.

        Returns:
            CoreSchema: The `pydantic-core` CoreSchema generated.
        """
        raise NotImplementedError

    def resolve_ref_schema(self, __maybe_ref_schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
        """Get the real schema for a `definition-ref` schema.
        If the schema given is not a `definition-ref` schema, it will be returned as is.
        This means you don't have to check before calling this function.

        Args:
            __maybe_ref_schema: A `CoreSchema`, `ref`-based or not.

        Raises:
            LookupError: If the `ref` is not found.

        Returns:
            A concrete `CoreSchema`.
        """
        raise NotImplementedError

    @property
    def field_name(self) -> str | None:
        """Get the name of the closest field to this validator."""
        raise NotImplementedError

    def _get_types_namespace(self) -> dict[str, Any] | None:
        """Internal method used during type resolution for serializer annotations."""
        raise NotImplementedError
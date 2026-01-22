from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import core_schema
from typing_extensions import Literal
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
class GenerateJsonSchemaHandler(GetJsonSchemaHandler):
    """JsonSchemaHandler implementation that doesn't do ref unwrapping by default.

    This is used for any Annotated metadata so that we don't end up with conflicting
    modifications to the definition schema.

    Used internally by Pydantic, please do not rely on this implementation.
    See `GetJsonSchemaHandler` for the handler API.
    """

    def __init__(self, generate_json_schema: GenerateJsonSchema, handler_override: HandlerOverride | None) -> None:
        self.generate_json_schema = generate_json_schema
        self.handler = handler_override or generate_json_schema.generate_inner
        self.mode = generate_json_schema.mode

    def __call__(self, __core_schema: CoreSchemaOrField) -> JsonSchemaValue:
        return self.handler(__core_schema)

    def resolve_ref_schema(self, maybe_ref_json_schema: JsonSchemaValue) -> JsonSchemaValue:
        """Resolves `$ref` in the json schema.

        This returns the input json schema if there is no `$ref` in json schema.

        Args:
            maybe_ref_json_schema: The input json schema that may contains `$ref`.

        Returns:
            Resolved json schema.

        Raises:
            LookupError: If it can't find the definition for `$ref`.
        """
        if '$ref' not in maybe_ref_json_schema:
            return maybe_ref_json_schema
        ref = maybe_ref_json_schema['$ref']
        json_schema = self.generate_json_schema.get_schema_from_definitions(ref)
        if json_schema is None:
            raise LookupError(f'Could not find a ref for {ref}. Maybe you tried to call resolve_ref_schema from within a recursive model?')
        return json_schema
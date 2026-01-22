from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import core_schema
from typing_extensions import Literal
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
class CallbackGetCoreSchemaHandler(GetCoreSchemaHandler):
    """Wrapper to use an arbitrary function as a `GetCoreSchemaHandler`.

    Used internally by Pydantic, please do not rely on this implementation.
    See `GetCoreSchemaHandler` for the handler API.
    """

    def __init__(self, handler: Callable[[Any], core_schema.CoreSchema], generate_schema: GenerateSchema, ref_mode: Literal['to-def', 'unpack']='to-def') -> None:
        self._handler = handler
        self._generate_schema = generate_schema
        self._ref_mode = ref_mode

    def __call__(self, __source_type: Any) -> core_schema.CoreSchema:
        schema = self._handler(__source_type)
        ref = schema.get('ref')
        if self._ref_mode == 'to-def':
            if ref is not None:
                self._generate_schema.defs.definitions[ref] = schema
                return core_schema.definition_reference_schema(ref)
            return schema
        else:
            return self.resolve_ref_schema(schema)

    def _get_types_namespace(self) -> dict[str, Any] | None:
        return self._generate_schema._types_namespace

    def generate_schema(self, __source_type: Any) -> core_schema.CoreSchema:
        return self._generate_schema.generate_schema(__source_type)

    @property
    def field_name(self) -> str | None:
        return self._generate_schema.field_name_stack.get()

    def resolve_ref_schema(self, maybe_ref_schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
        """Resolves reference in the core schema.

        Args:
            maybe_ref_schema: The input core schema that may contains reference.

        Returns:
            Resolved core schema.

        Raises:
            LookupError: If it can't find the definition for reference.
        """
        if maybe_ref_schema['type'] == 'definition-ref':
            ref = maybe_ref_schema['schema_ref']
            if ref not in self._generate_schema.defs.definitions:
                raise LookupError(f'Could not find a ref for {ref}. Maybe you tried to call resolve_ref_schema from within a recursive model?')
            return self._generate_schema.defs.definitions[ref]
        elif maybe_ref_schema['type'] == 'definitions':
            return self.resolve_ref_schema(maybe_ref_schema['schema'])
        return maybe_ref_schema
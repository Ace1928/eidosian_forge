from __future__ import annotations
from typing import Any, Callable, NamedTuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Literal, Protocol, TypeAlias
class ValidatePythonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    """Event handler for `SchemaValidator.validate_python`."""

    def on_enter(self, input: Any, *, strict: bool | None=None, from_attributes: bool | None=None, context: dict[str, Any] | None=None, self_instance: Any | None=None) -> None:
        """Callback to be notified of validation start, and create an instance of the event handler.

        Args:
            input: The input to be validated.
            strict: Whether to validate the object in strict mode.
            from_attributes: Whether to validate objects as inputs by extracting attributes.
            context: The context to use for validation, this is passed to functional validators.
            self_instance: An instance of a model to set attributes on from validation, this is used when running
                validation from the `__init__` method of a model.
        """
        pass
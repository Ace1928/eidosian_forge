from __future__ import annotations
from typing import Any, Callable, NamedTuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Literal, Protocol, TypeAlias
class BaseValidateHandlerProtocol(Protocol):
    """Base class for plugin callbacks protocols.

    You shouldn't implement this protocol directly, instead use one of the subclasses with adds the correctly
    typed `on_error` method.
    """
    on_enter: Callable[..., None]
    '`on_enter` is changed to be more specific on all subclasses'

    def on_success(self, result: Any) -> None:
        """Callback to be notified of successful validation.

        Args:
            result: The result of the validation.
        """
        return

    def on_error(self, error: ValidationError) -> None:
        """Callback to be notified of validation errors.

        Args:
            error: The validation error.
        """
        return

    def on_exception(self, exception: Exception) -> None:
        """Callback to be notified of validation exceptions.

        Args:
            exception: The exception raised during validation.
        """
        return
from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from .handler import Handler
 Processes incoming HTTP request returning a dictionary of
        additional data to add to the session_context.

        Args:
            request: HTTP request

        Returns:
            A dictionary of JSON serializable data to be included on
            the session context.
        
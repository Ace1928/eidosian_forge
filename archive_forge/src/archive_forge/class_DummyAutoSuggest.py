from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.eventloop import run_in_executor_with_context
from .document import Document
from .filters import Filter, to_filter
class DummyAutoSuggest(AutoSuggest):
    """
    AutoSuggest class that doesn't return any suggestion.
    """

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
        return None
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.eventloop import run_in_executor_with_context
from .document import Document
from .filters import Filter, to_filter
class DynamicAutoSuggest(AutoSuggest):
    """
    Validator class that can dynamically returns any Validator.

    :param get_validator: Callable that returns a :class:`.Validator` instance.
    """

    def __init__(self, get_auto_suggest: Callable[[], AutoSuggest | None]) -> None:
        self.get_auto_suggest = get_auto_suggest

    def get_suggestion(self, buff: Buffer, document: Document) -> Suggestion | None:
        auto_suggest = self.get_auto_suggest() or DummyAutoSuggest()
        return auto_suggest.get_suggestion(buff, document)

    async def get_suggestion_async(self, buff: Buffer, document: Document) -> Suggestion | None:
        auto_suggest = self.get_auto_suggest() or DummyAutoSuggest()
        return await auto_suggest.get_suggestion_async(buff, document)
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, Callable, Iterable, Sequence
from prompt_toolkit.document import Document
from prompt_toolkit.eventloop import aclosing, generator_to_async_generator
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
class DynamicCompleter(Completer):
    """
    Completer class that can dynamically returns any Completer.

    :param get_completer: Callable that returns a :class:`.Completer` instance.
    """

    def __init__(self, get_completer: Callable[[], Completer | None]) -> None:
        self.get_completer = get_completer

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        completer = self.get_completer() or DummyCompleter()
        return completer.get_completions(document, complete_event)

    async def get_completions_async(self, document: Document, complete_event: CompleteEvent) -> AsyncGenerator[Completion, None]:
        completer = self.get_completer() or DummyCompleter()
        async for completion in completer.get_completions_async(document, complete_event):
            yield completion

    def __repr__(self) -> str:
        return f'DynamicCompleter({self.get_completer!r} -> {self.get_completer()!r})'
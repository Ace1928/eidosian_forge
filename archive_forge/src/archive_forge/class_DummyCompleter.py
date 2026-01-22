from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, Callable, Iterable, Sequence
from prompt_toolkit.document import Document
from prompt_toolkit.eventloop import aclosing, generator_to_async_generator
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
class DummyCompleter(Completer):
    """
    A completer that doesn't return any completion.
    """

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        return []

    def __repr__(self) -> str:
        return 'DummyCompleter()'
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, Callable, Iterable, Sequence
from prompt_toolkit.document import Document
from prompt_toolkit.eventloop import aclosing, generator_to_async_generator
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
class _MergedCompleter(Completer):
    """
    Combine several completers into one.
    """

    def __init__(self, completers: Sequence[Completer]) -> None:
        self.completers = completers

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)

    async def get_completions_async(self, document: Document, complete_event: CompleteEvent) -> AsyncGenerator[Completion, None]:
        for completer in self.completers:
            async with aclosing(completer.get_completions_async(document, complete_event)) as async_generator:
                async for item in async_generator:
                    yield item
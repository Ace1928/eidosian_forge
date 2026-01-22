from __future__ import annotations
from collections.abc import MutableMapping as MutableMappingABC
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping, TypedDict, cast
class OptionsDict(MutableMappingABC):
    """A dictionary, with attribute access to core markdownit configuration options."""

    def __init__(self, options: OptionsType) -> None:
        self._options = cast(OptionsType, dict(options))

    def __getitem__(self, key: str) -> Any:
        return self._options[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._options[key] = value

    def __delitem__(self, key: str) -> None:
        del self._options[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._options)

    def __len__(self) -> int:
        return len(self._options)

    def __repr__(self) -> str:
        return repr(self._options)

    def __str__(self) -> str:
        return str(self._options)

    @property
    def maxNesting(self) -> int:
        """Internal protection, recursion limit."""
        return self._options['maxNesting']

    @maxNesting.setter
    def maxNesting(self, value: int) -> None:
        self._options['maxNesting'] = value

    @property
    def html(self) -> bool:
        """Enable HTML tags in source."""
        return self._options['html']

    @html.setter
    def html(self, value: bool) -> None:
        self._options['html'] = value

    @property
    def linkify(self) -> bool:
        """Enable autoconversion of URL-like texts to links."""
        return self._options['linkify']

    @linkify.setter
    def linkify(self, value: bool) -> None:
        self._options['linkify'] = value

    @property
    def typographer(self) -> bool:
        """Enable smartquotes and replacements."""
        return self._options['typographer']

    @typographer.setter
    def typographer(self, value: bool) -> None:
        self._options['typographer'] = value

    @property
    def quotes(self) -> str:
        """Quote characters."""
        return self._options['quotes']

    @quotes.setter
    def quotes(self, value: str) -> None:
        self._options['quotes'] = value

    @property
    def xhtmlOut(self) -> bool:
        """Use '/' to close single tags (<br />)."""
        return self._options['xhtmlOut']

    @xhtmlOut.setter
    def xhtmlOut(self, value: bool) -> None:
        self._options['xhtmlOut'] = value

    @property
    def breaks(self) -> bool:
        """Convert newlines in paragraphs into <br>."""
        return self._options['breaks']

    @breaks.setter
    def breaks(self, value: bool) -> None:
        self._options['breaks'] = value

    @property
    def langPrefix(self) -> str:
        """CSS language prefix for fenced blocks."""
        return self._options['langPrefix']

    @langPrefix.setter
    def langPrefix(self, value: str) -> None:
        self._options['langPrefix'] = value

    @property
    def highlight(self) -> Callable[[str, str, str], str] | None:
        """Highlighter function: (content, langName, langAttrs) -> escaped HTML."""
        return self._options['highlight']

    @highlight.setter
    def highlight(self, value: Callable[[str, str, str], str] | None) -> None:
        self._options['highlight'] = value
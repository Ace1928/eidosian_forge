import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class _KVSection(Section):
    """Base parser for numpydoc sections with key-value syntax.

    E.g. sections that look like this:
        key
            value
        key2 : type
            values can also span...
            ... multiple lines
    """

    def _parse_item(self, key: str, value: str) -> DocstringMeta:
        pass

    def parse(self, text: str) -> T.Iterable[DocstringMeta]:
        for match, next_match in _pairwise(KV_REGEX.finditer(text)):
            start = match.end()
            end = next_match.start() if next_match is not None else None
            value = text[start:end]
            yield self._parse_item(key=match.group(), value=inspect.cleandoc(value))
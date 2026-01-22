from __future__ import annotations
from collections.abc import MutableMapping as MutableMappingABC
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping, TypedDict, cast
class OptionsType(TypedDict):
    """Options for parsing."""
    maxNesting: int
    'Internal protection, recursion limit.'
    html: bool
    'Enable HTML tags in source.'
    linkify: bool
    'Enable autoconversion of URL-like texts to links.'
    typographer: bool
    'Enable smartquotes and replacements.'
    quotes: str
    'Quote characters.'
    xhtmlOut: bool
    "Use '/' to close single tags (<br />)."
    breaks: bool
    'Convert newlines in paragraphs into <br>.'
    langPrefix: str
    'CSS language prefix for fenced blocks.'
    highlight: Callable[[str, str, str], str] | None
    'Highlighter function: (content, lang, attrs) -> str.'
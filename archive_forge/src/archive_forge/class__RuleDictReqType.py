from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
class _RuleDictReqType(TypedDict):
    name: str
    rex: re.Pattern[str]
    tmpl: str
    tag: str
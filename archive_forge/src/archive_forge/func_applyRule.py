from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
def applyRule(rule: RuleDictType, string: str, begin: int, inBlockquote: bool) -> None | Match[str]:
    if not (string.startswith(rule['tag'], begin) and (rule['pre'](string, begin) if 'pre' in rule else True)):
        return None
    match = rule['rex'].match(string[begin:])
    if not match or match.start() != 0:
        return None
    lastIndex = match.end() + begin - 1
    if 'post' in rule and (not (rule['post'](string, lastIndex) and (not inBlockquote or '\n' not in match.group(1)))):
        return None
    return match
import re
from abc import ABC, abstractmethod
from typing import List, Union
from .text import Span, Text
class JSONHighlighter(RegexHighlighter):
    """Highlights JSON"""
    JSON_STR = '(?<![\\\\\\w])(?P<str>b?\\".*?(?<!\\\\)\\")'
    JSON_WHITESPACE = {' ', '\n', '\r', '\t'}
    base_style = 'json.'
    highlights = [_combine_regex('(?P<brace>[\\{\\[\\(\\)\\]\\}])', '\\b(?P<bool_true>true)\\b|\\b(?P<bool_false>false)\\b|\\b(?P<null>null)\\b', '(?P<number>(?<!\\w)\\-?[0-9]+\\.?[0-9]*(e[\\-\\+]?\\d+?)?\\b|0x[0-9a-fA-F]*)', JSON_STR)]

    def highlight(self, text: Text) -> None:
        super().highlight(text)
        plain = text.plain
        append = text.spans.append
        whitespace = self.JSON_WHITESPACE
        for match in re.finditer(self.JSON_STR, plain):
            start, end = match.span()
            cursor = end
            while cursor < len(plain):
                char = plain[cursor]
                cursor += 1
                if char == ':':
                    append(Span(start, end, 'json.key'))
                elif char in whitespace:
                    continue
                break
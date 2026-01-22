import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def _get_fragments(txt: str, pattern: Pattern[str]) -> Iterable[Union[str, Match[str]]]:
    offset = 0
    for match in pattern.finditer(txt):
        match_s, match_e = match.span(1)
        yield txt[offset:match_s]
        yield match
        offset = match_e
    yield txt[offset:]
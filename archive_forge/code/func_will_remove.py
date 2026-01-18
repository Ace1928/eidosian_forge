import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def will_remove(tag: str) -> bool:
    tag = tag.lower()
    if which_ones:
        return tag in which_ones
    else:
        return tag not in keep
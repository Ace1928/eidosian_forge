import re
from typing import Iterator, Optional
from .rio import Stanza
def _valid_tag(tag: str) -> bool:
    if not isinstance(tag, str):
        raise TypeError(tag)
    return bool(_tag_re.match(tag))
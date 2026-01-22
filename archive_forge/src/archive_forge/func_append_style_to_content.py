from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.cache import FastDictCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.utils import get_cwidth
def append_style_to_content(self, style_str: str) -> None:
    """
        For all the characters in the screen.
        Set the style string to the given `style_str`.
        """
    b = self.data_buffer
    char_cache = _CHAR_CACHE
    append_style = ' ' + style_str
    for y, row in b.items():
        for x, char in row.items():
            row[x] = char_cache[char.char, char.style + append_style]
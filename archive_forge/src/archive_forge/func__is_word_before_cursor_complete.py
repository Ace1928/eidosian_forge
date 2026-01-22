from __future__ import annotations
import bisect
import re
import string
import weakref
from typing import Callable, Dict, Iterable, List, NoReturn, Pattern, cast
from .clipboard import ClipboardData
from .filters import vi_mode
from .selection import PasteMode, SelectionState, SelectionType
def _is_word_before_cursor_complete(self, WORD: bool=False, pattern: Pattern[str] | None=None) -> bool:
    if pattern:
        return self.find_start_of_previous_word(WORD=WORD, pattern=pattern) is None
    else:
        return self.text_before_cursor == '' or self.text_before_cursor[-1:].isspace()
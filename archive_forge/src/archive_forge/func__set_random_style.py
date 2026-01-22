from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _set_random_style(self) -> None:
    import random
    self.header = random.choice((True, False))
    self.border = random.choice((True, False))
    self._hrules = random.choice((ALL, FRAME, HEADER, NONE))
    self._vrules = random.choice((ALL, FRAME, NONE))
    self.left_padding_width = random.randint(0, 5)
    self.right_padding_width = random.randint(0, 5)
    self.vertical_char = random.choice('~!@#$%^&*()_+|-=\\{}[];\':\\",./;<>?')
    self.horizontal_char = random.choice('~!@#$%^&*()_+|-=\\{}[];\':\\",./;<>?')
    self.junction_char = random.choice('~!@#$%^&*()_+|-=\\{}[];\':\\",./;<>?')
    self.preserve_internal_border = random.choice((True, False))
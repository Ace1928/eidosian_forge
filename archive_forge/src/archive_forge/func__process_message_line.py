from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _process_message_line(self, lineno, line, obsolete=False) -> None:
    if line.startswith('"'):
        self._process_string_continuation_line(line, lineno)
    else:
        self._process_keyword_line(lineno, line, obsolete)
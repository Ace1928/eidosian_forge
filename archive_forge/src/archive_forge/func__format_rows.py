from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _format_rows(self, rows):
    return [self._format_row(row) for row in rows]
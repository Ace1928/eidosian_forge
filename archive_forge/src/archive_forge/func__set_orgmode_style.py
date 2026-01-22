from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _set_orgmode_style(self) -> None:
    self._set_default_style()
    self.orgmode = True
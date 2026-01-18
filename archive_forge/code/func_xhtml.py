from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@xhtml.setter
def xhtml(self, val) -> None:
    self._validate_option('xhtml', val)
    self._xhtml = val
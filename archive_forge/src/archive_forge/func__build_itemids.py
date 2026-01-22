from __future__ import annotations
import collections
from functools import partial
from typing import Any, Set
import html_text
import lxml.etree
from lxml.html.clean import Cleaner
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def _build_itemids(self, document):
    """Build itemids for a fast get_docid implementation. Use document order."""
    root = document.getroottree().getroot()
    return {node: idx + 1 for idx, node in enumerate(self._xp_item(root))}
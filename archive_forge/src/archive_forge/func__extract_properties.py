from __future__ import annotations
import collections
from functools import partial
from typing import Any, Set
import html_text
import lxml.etree
from lxml.html.clean import Cleaner
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def _extract_properties(self, node, items_seen, base_url, itemids):
    for prop in self._xp_prop(node):
        for p, v in self._extract_property(prop, items_seen=items_seen, base_url=base_url, itemids=itemids):
            yield (p, v)
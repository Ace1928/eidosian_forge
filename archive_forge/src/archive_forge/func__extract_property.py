from __future__ import annotations
import collections
from functools import partial
from typing import Any, Set
import html_text
import lxml.etree
from lxml.html.clean import Cleaner
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def _extract_property(self, node, items_seen, base_url, itemids):
    props = node.get('itemprop').split()
    value = self._extract_property_value(node, items_seen=items_seen, base_url=base_url, itemids=itemids)
    return [(p, value) for p in props]
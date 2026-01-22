from __future__ import annotations
import collections
from functools import partial
from typing import Any, Set
import html_text
import lxml.etree
from lxml.html.clean import Cleaner
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def _extract_property_refs(self, node, refid, items_seen, base_url, itemids):
    ref_node = node.xpath('id($refid)[1]', refid=refid)
    if not ref_node:
        return
    ref_node = ref_node[0]
    extract_fn = partial(self._extract_property, items_seen=items_seen, base_url=base_url, itemids=itemids)
    if 'itemprop' in ref_node.keys() and 'itemscope' in ref_node.keys():
        for p, v in extract_fn(ref_node):
            yield (p, v)
    else:
        base_parent_scope = ref_node.xpath('ancestor-or-self::*[@itemscope][1]')
        for prop in ref_node.xpath('descendant-or-self::*[@itemprop]'):
            parent_scope = prop.xpath('ancestor::*[@itemscope][1]')
            if parent_scope == base_parent_scope:
                for p, v in extract_fn(prop):
                    yield (p, v)
from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING, Any, Sequence
def _build_empty_row(self, parent: etree.Element, align: Sequence[str | None]) -> None:
    """Build an empty row."""
    tr = etree.SubElement(parent, 'tr')
    count = len(align)
    while count:
        etree.SubElement(tr, 'td')
        count -= 1
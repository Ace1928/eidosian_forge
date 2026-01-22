from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING, Any, Sequence
def _split_row(self, row: str) -> list[str]:
    """ split a row of text into list of cells. """
    if self.border:
        if row.startswith('|'):
            row = row[1:]
        row = self.RE_END_BORDER.sub('', row)
    return self._split(row)
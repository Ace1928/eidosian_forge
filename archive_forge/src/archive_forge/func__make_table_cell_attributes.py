from __future__ import annotations
from collections import defaultdict
import datetime
import json
from typing import (
from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (
def _make_table_cell_attributes(self, cell) -> dict[str, int | str]:
    """Convert cell attributes to OpenDocument attributes

        Parameters
        ----------
        cell : ExcelCell
            Spreadsheet cell data

        Returns
        -------
        attributes : Dict[str, Union[int, str]]
            Dictionary with attributes and attribute values
        """
    attributes: dict[str, int | str] = {}
    style_name = self._process_style(cell.style)
    if style_name is not None:
        attributes['stylename'] = style_name
    if cell.mergestart is not None and cell.mergeend is not None:
        attributes['numberrowsspanned'] = max(1, cell.mergestart)
        attributes['numbercolumnsspanned'] = cell.mergeend
    return attributes
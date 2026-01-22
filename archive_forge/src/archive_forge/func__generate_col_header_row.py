from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def _generate_col_header_row(self, iter: Sequence, max_cols: int, col_lengths: dict):
    """
        Generate the row containing column headers:

         +----------------------------+---------------+---------------------------+
         |  index_blanks ...          | column_name_i |  column_headers (level_i) |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns
        col_lengths :
            c

        Returns
        -------
        list of elements
        """
    r, clabels = iter
    index_blanks = [_element('th', self.css['blank'], self.css['blank_value'], True)] * (self.index.nlevels - sum(self.hide_index_) - 1)
    name = self.data.columns.names[r]
    column_name = [_element('th', f'{self.css['blank']} {self.css['level']}{r}' if name is None else f'{self.css['index_name']} {self.css['level']}{r}', name if name is not None and (not self.hide_column_names) else self.css['blank_value'], not all(self.hide_index_))]
    column_headers: list = []
    visible_col_count: int = 0
    for c, value in enumerate(clabels[r]):
        header_element_visible = _is_visible(c, r, col_lengths)
        if header_element_visible:
            visible_col_count += col_lengths.get((r, c), 0)
        if self._check_trim(visible_col_count, max_cols, column_headers, 'th', f'{self.css['col_heading']} {self.css['level']}{r} {self.css['col_trim']}'):
            break
        header_element = _element('th', f'{self.css['col_heading']} {self.css['level']}{r} {self.css['col']}{c}', value, header_element_visible, display_value=self._display_funcs_columns[r, c](value), attributes=f'colspan="{col_lengths.get((r, c), 0)}"' if col_lengths.get((r, c), 0) > 1 else '')
        if self.cell_ids:
            header_element['id'] = f'{self.css['level']}{r}_{self.css['col']}{c}'
        if header_element_visible and (r, c) in self.ctx_columns and self.ctx_columns[r, c]:
            header_element['id'] = f'{self.css['level']}{r}_{self.css['col']}{c}'
            self.cellstyle_map_columns[tuple(self.ctx_columns[r, c])].append(f'{self.css['level']}{r}_{self.css['col']}{c}')
        column_headers.append(header_element)
    return index_blanks + column_name + column_headers
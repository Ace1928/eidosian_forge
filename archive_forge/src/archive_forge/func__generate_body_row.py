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
def _generate_body_row(self, iter: tuple, max_cols: int, idx_lengths: dict):
    """
        Generate a regular row for the body section of appropriate format.

          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Iterable from outer scope: row number, row data tuple, row index labels.
        max_cols : int
            Number of permissible columns.
        idx_lengths : dict
            A map of the sparsification structure of the index

        Returns
        -------
            list of elements
        """
    r, row_tup, rlabels = iter
    index_headers = []
    for c, value in enumerate(rlabels[r]):
        header_element_visible = _is_visible(r, c, idx_lengths) and (not self.hide_index_[c])
        header_element = _element('th', f'{self.css['row_heading']} {self.css['level']}{c} {self.css['row']}{r}', value, header_element_visible, display_value=self._display_funcs_index[r, c](value), attributes=f'rowspan="{idx_lengths.get((c, r), 0)}"' if idx_lengths.get((c, r), 0) > 1 else '')
        if self.cell_ids:
            header_element['id'] = f'{self.css['level']}{c}_{self.css['row']}{r}'
        if header_element_visible and (r, c) in self.ctx_index and self.ctx_index[r, c]:
            header_element['id'] = f'{self.css['level']}{c}_{self.css['row']}{r}'
            self.cellstyle_map_index[tuple(self.ctx_index[r, c])].append(f'{self.css['level']}{c}_{self.css['row']}{r}')
        index_headers.append(header_element)
    data: list = []
    visible_col_count: int = 0
    for c, value in enumerate(row_tup[1:]):
        data_element_visible = c not in self.hidden_columns and r not in self.hidden_rows
        if data_element_visible:
            visible_col_count += 1
        if self._check_trim(visible_col_count, max_cols, data, 'td', f'{self.css['data']} {self.css['row']}{r} {self.css['col_trim']}'):
            break
        cls = ''
        if (r, c) in self.cell_context:
            cls = ' ' + self.cell_context[r, c]
        data_element = _element('td', f'{self.css['data']} {self.css['row']}{r} {self.css['col']}{c}{cls}', value, data_element_visible, attributes='', display_value=self._display_funcs[r, c](value))
        if self.cell_ids:
            data_element['id'] = f'{self.css['row']}{r}_{self.css['col']}{c}'
        if data_element_visible and (r, c) in self.ctx and self.ctx[r, c]:
            data_element['id'] = f'{self.css['row']}{r}_{self.css['col']}{c}'
            self.cellstyle_map[tuple(self.ctx[r, c])].append(f'{self.css['row']}{r}_{self.css['col']}{c}')
        data.append(data_element)
    return index_headers + data
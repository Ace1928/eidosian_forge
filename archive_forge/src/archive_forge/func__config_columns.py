from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
def _config_columns(self, column_objs: List[TableColumn]) -> List[Dict[str, Any]]:
    column_objs = list(column_objs)
    groups = {}
    columns = []
    selectable = self.selectable
    if self.row_content:
        columns.append({'formatter': 'expand'})
    if isinstance(selectable, str) and selectable.startswith('checkbox'):
        title = '' if selectable.endswith('-single') else 'rowSelection'
        columns.append({'formatter': 'rowSelection', 'titleFormatter': title, 'hozAlign': 'center', 'headerSort': False, 'frozen': True, 'width': 40})
    if isinstance(self.frozen_columns, dict):
        left_frozen_columns = [col for col in column_objs if self.frozen_columns.get(col.field, self.frozen_columns.get(column_objs.index(col))) == 'left']
        right_frozen_columns = [col for col in column_objs if self.frozen_columns.get(col.field, self.frozen_columns.get(column_objs.index(col))) == 'right']
        non_frozen_columns = [col for col in column_objs if col.field not in self.frozen_columns and column_objs.index(col) not in self.frozen_columns]
        ordered_columns = left_frozen_columns + non_frozen_columns + right_frozen_columns
    else:
        ordered_columns = []
        for col in self.frozen_columns:
            if isinstance(col, int):
                ordered_columns.append(column_objs.pop(col))
            else:
                cols = [c for c in column_objs if c.field == col]
                if cols:
                    ordered_columns.append(cols[0])
                    column_objs.remove(cols[0])
        ordered_columns += column_objs
    grouping = {group: [str(gc) for gc in group_cols] for group, group_cols in self.groups.items()}
    for i, column in enumerate(ordered_columns):
        field = column.field
        matching_groups = [group for group, group_cols in grouping.items() if field in group_cols]
        col_dict = dict(field=field)
        if isinstance(self.sortable, dict):
            col_dict['headerSort'] = self.sortable.get(field, True)
        elif not self.sortable:
            col_dict['headerSort'] = self.sortable
        if isinstance(self.text_align, str):
            col_dict['hozAlign'] = self.text_align
        elif field in self.text_align:
            col_dict['hozAlign'] = self.text_align[field]
        if isinstance(self.header_align, str):
            col_dict['headerHozAlign'] = self.header_align
        elif field in self.header_align:
            col_dict['headerHozAlign'] = self.header_align[field]
        formatter = self.formatters.get(field)
        if isinstance(formatter, str):
            col_dict['formatter'] = formatter
        elif isinstance(formatter, dict):
            formatter = dict(formatter)
            col_dict['formatter'] = formatter.pop('type')
            col_dict['formatterParams'] = formatter
        title_formatter = self.title_formatters.get(field)
        if isinstance(title_formatter, str):
            col_dict['titleFormatter'] = title_formatter
        elif isinstance(title_formatter, dict):
            title_formatter = dict(title_formatter)
            col_dict['titleFormatter'] = title_formatter.pop('type')
            col_dict['titleFormatterParams'] = title_formatter
        col_name = self._renamed_cols[field]
        if field in self.indexes:
            if len(self.indexes) == 1:
                dtype = self.value.index.dtype
            else:
                dtype = self.value.index.get_level_values(self.indexes.index(field)).dtype
        else:
            dtype = self.value.dtypes[col_name]
        if dtype.kind == 'M':
            col_dict['sorter'] = 'timestamp'
        elif dtype.kind in 'iuf':
            col_dict['sorter'] = 'number'
        elif dtype.kind == 'b':
            col_dict['sorter'] = 'boolean'
        editor = self.editors.get(field)
        if field in self.editors and editor is None:
            col_dict['editable'] = False
        if isinstance(editor, str):
            col_dict['editor'] = editor
        elif isinstance(editor, dict):
            editor = dict(editor)
            col_dict['editor'] = editor.pop('type')
            col_dict['editorParams'] = editor
        if col_dict.get('editor') in ['select', 'autocomplete']:
            self.param.warning(f'The {col_dict['editor']!r} editor has been deprecated, use instead the "list" editor type to configure column {field!r}')
            col_dict['editor'] = 'list'
            if col_dict.get('editorParams', {}).get('values', False) is True:
                del col_dict['editorParams']['values']
                col_dict['editorParams']['valuesLookup'] = True
        if field in self.frozen_columns or i in self.frozen_columns:
            col_dict['frozen'] = True
        if isinstance(self.widths, dict) and isinstance(self.widths.get(field), str):
            col_dict['width'] = self.widths[field]
        col_dict.update(self._get_filter_spec(column))
        if matching_groups:
            group = matching_groups[0]
            if group in groups:
                groups[group]['columns'].append(col_dict)
                continue
            group_dict = {'title': group, 'columns': [col_dict]}
            groups[group] = group_dict
            columns.append(group_dict)
        else:
            columns.append(col_dict)
    return columns
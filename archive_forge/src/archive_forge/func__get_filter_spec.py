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
def _get_filter_spec(self, column: TableColumn) -> Dict[str, Any]:
    fspec = {}
    if not self.header_filters or (isinstance(self.header_filters, dict) and column.field not in self.header_filters):
        return fspec
    elif self.header_filters == True:
        if column.field in self.indexes:
            if len(self.indexes) == 1:
                col = self.value.index
            else:
                col = self.value.index.get_level_values(self.indexes.index(column.field))
            if col.dtype.kind in 'uif':
                fspec['headerFilter'] = 'number'
            elif col.dtype.kind == 'b':
                fspec['headerFilter'] = 'tickCross'
                fspec['headerFilterParams'] = {'tristate': True, 'indeterminateValue': None}
            elif isdatetime(col) or col.dtype.kind == 'M':
                fspec['headerFilter'] = False
            else:
                fspec['headerFilter'] = True
        elif isinstance(column.editor, DateEditor):
            fspec['headerFilter'] = False
        else:
            fspec['headerFilter'] = True
        return fspec
    filter_type = self.header_filters[column.field]
    if isinstance(filter_type, dict):
        filter_params = dict(filter_type)
        filter_type = filter_params.pop('type', True)
        filter_func = filter_params.pop('func', None)
        filter_placeholder = filter_params.pop('placeholder', None)
    else:
        filter_params = {}
        filter_func = None
        filter_placeholder = None
    if filter_type in ['select', 'autocomplete']:
        self.param.warning(f'The {filter_type!r} filter has been deprecated, use instead the "list" filter type to configure column {column.field!r}')
        filter_type = 'list'
        if filter_params.get('values', False) is True:
            self.param.warning(f'Setting "values" to True has been deprecated, instead set "valuesLookup" to True to configure column {column.field!r}')
            del filter_params['values']
            filter_params['valuesLookup'] = True
    if filter_type == 'list' and (not filter_params):
        filter_params = {'valuesLookup': True}
    fspec['headerFilter'] = filter_type
    if filter_params:
        fspec['headerFilterParams'] = filter_params
    if filter_func:
        fspec['headerFilterFunc'] = filter_func
    if filter_placeholder:
        fspec['headerFilterPlaceholder'] = filter_placeholder
    return fspec
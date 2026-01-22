import os
import numpy as np
import pandas.json as json
from pandas.tslib import iNaT
from pandas.compat import StringIO, long, u
from pandas import compat, isnull
from pandas import Series, DataFrame, to_datetime, MultiIndex
from pandas.io.common import (get_filepath_or_buffer, _get_handle,
from pandas.core.common import AbstractMethodError
from pandas.formats.printing import pprint_thing
from .normalize import _convert_to_line_delimits
from .table_schema import build_table_schema
class JSONTableWriter(FrameWriter):
    _default_orient = 'records'

    def __init__(self, obj, orient, date_format, double_precision, ensure_ascii, date_unit, default_handler=None):
        """
        Adds a `schema` attribut with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        super(JSONTableWriter, self).__init__(obj, orient, date_format, double_precision, ensure_ascii, date_unit, default_handler=default_handler)
        if date_format != 'iso':
            msg = "Trying to write with `orient='table'` and `date_format='%s'`. Table Schema requires dates to be formatted with `date_format='iso'`" % date_format
            raise ValueError(msg)
        self.schema = build_table_schema(obj)
        if obj.ndim == 2 and isinstance(obj.columns, MultiIndex):
            raise NotImplementedError("orient='table' is not supported for MultiIndex")
        if obj.ndim == 1 and obj.name in set(obj.index.names) or len(obj.columns & obj.index.names):
            msg = 'Overlapping names between the index and columns'
            raise ValueError(msg)
        obj = obj.copy()
        timedeltas = obj.select_dtypes(include=['timedelta']).columns
        if len(timedeltas):
            obj[timedeltas] = obj[timedeltas].applymap(lambda x: x.isoformat())
        self.obj = obj.reset_index()
        self.date_format = 'iso'
        self.orient = 'records'

    def write(self):
        data = super(JSONTableWriter, self).write()
        serialized = '{{"schema": {}, "data": {}}}'.format(dumps(self.schema), data)
        return serialized
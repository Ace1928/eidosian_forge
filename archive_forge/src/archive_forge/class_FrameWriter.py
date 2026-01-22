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
class FrameWriter(Writer):
    _default_orient = 'columns'

    def _format_axes(self):
        """ try to axes if they are datelike """
        if not self.obj.index.is_unique and self.orient in ('index', 'columns'):
            raise ValueError("DataFrame index must be unique for orient='%s'." % self.orient)
        if not self.obj.columns.is_unique and self.orient in ('index', 'columns', 'records'):
            raise ValueError("DataFrame columns must be unique for orient='%s'." % self.orient)
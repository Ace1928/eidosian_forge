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
def _process_df_and_convert_to_cds(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataDict]:
    import pandas as pd
    df = self._filter_dataframe(df)
    if df is None:
        return ([], {})
    if isinstance(self.value.index, pd.MultiIndex):
        indexes = [f'level_{i}' if n is None else n for i, n in enumerate(df.index.names)]
    else:
        default_index = 'level_0' if 'index' in df.columns else 'index'
        indexes = [df.index.name or default_index]
    if len(indexes) > 1:
        df = df.reset_index()
    data = ColumnDataSource.from_df(df)
    if not self.show_index and len(indexes) > 1:
        data = {k: v for k, v in data.items() if k not in indexes}
    return (df, {k if isinstance(k, str) else str(k): self._process_column(v) for k, v in data.items()})
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
def _get_configuration(self, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
        Returns the Tabulator configuration.
        """
    configuration = dict(self._configuration)
    if 'selectable' not in configuration:
        configuration['selectable'] = self.selectable
    if self.groups and 'columns' in configuration:
        raise ValueError('Groups must be defined either explicitly or via the configuration, not both.')
    configuration['columns'] = self._config_columns(columns)
    configuration['dataTree'] = self.hierarchical
    if self.sizing_mode in ('stretch_height', 'stretch_both'):
        configuration['maxHeight'] = '100%'
    elif self.height:
        configuration['height'] = self.height
    return configuration
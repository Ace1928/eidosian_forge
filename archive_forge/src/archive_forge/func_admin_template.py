import datetime as dt
import logging
import os
import sys
import time
from functools import partial
import bokeh
import numpy as np
import pandas as pd
import param
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from ..config import config, panel_extension as extension
from ..depends import bind
from ..layout import (
from ..pane import HTML, Bokeh
from ..template import FastListTemplate
from ..widgets import (
from ..widgets.indicators import Trend
from .logging import (
from .notebook import push_notebook
from .profile import profiling_tabs
from .server import set_curdoc
from .state import state
def admin_template(doc):
    extension('tabulator', 'terminal')
    log_sessions.append(id(doc))

    def _remove_log_session(session_context):
        log_sessions.remove(id(doc))
    doc.on_session_destroyed(_remove_log_session)
    template = FastListTemplate(title='Admin Panel', theme='dark')
    tabs = Tabs(('Overview', get_overview(doc)), ('Timeline', get_timeline(doc)), margin=0, sizing_mode='stretch_both')
    if config.profiler:
        tabs.append(('Launch Profiling', profiling_tabs(state, '^\\/.*', None)))
    tabs.extend([('User Profiling', profiling_tabs(state, None, '^\\/.*')), ('Logs', log_component())])
    tabs.extend([(name, plugin()) for name, plugin in config.admin_plugins])
    template.main.append(tabs)
    return template
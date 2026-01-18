import asyncio
import os
import pandas as pd
import param
import pytest
from bokeh.models import (
from packaging.version import Version
from panel import config
from panel.depends import bind
from panel.io.state import set_curdoc, state
from panel.layout import Row, Tabs
from panel.models import HTML as BkHTML
from panel.pane import (
from panel.param import (
from panel.tests.util import mpl_available, mpl_figure
from panel.widgets import (
def test_paramfunction_bare_lazy_no_warning(caplog):

    def foo():
        return 'bar'
    ParamFunction(foo, lazy=True)
    for log_record in caplog.records:
        assert "The function 'foo' does not have any dependencies and will never update" not in log_record.message
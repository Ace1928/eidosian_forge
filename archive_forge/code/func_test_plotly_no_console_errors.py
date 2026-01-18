import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
def test_plotly_no_console_errors(page, plotly_2d_plot):
    msgs, _ = serve_component(page, plotly_2d_plot)
    page.wait_for_timeout(1000)
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []
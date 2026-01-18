import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
def test_plotly_click_data(page, plotly_2d_plot):
    serve_component(page, plotly_2d_plot)
    plotly_plot = page.locator('.js-plotly-plot .plot-container.plotly')
    expect(plotly_plot).to_have_count(1)
    point = page.locator('.js-plotly-plot .plot-container.plotly path.point').nth(0)
    point.click(force=True)
    wait_until(lambda: plotly_2d_plot.click_data == {'points': [{'curveNumber': 0, 'pointIndex': 0, 'pointNumber': 0, 'x': 0, 'y': 2}]}, page)
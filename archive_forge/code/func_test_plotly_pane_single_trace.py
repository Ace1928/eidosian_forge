import datetime as dt
import pytest
import numpy as np
import pandas as pd
from panel.models.plotly import PlotlyPlot
from panel.pane import PaneBase, Plotly
@plotly_available
def test_plotly_pane_single_trace(document, comm):
    trace = go.Scatter(x=[0, 1], y=[2, 3], uid='Test')
    pane = Plotly({'data': [trace], 'layout': {'width': 350}})
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, PlotlyPlot)
    assert pane._models[model.ref['id']][0] is model
    assert len(model.data) == 1
    assert model.data[0]['type'] == 'scatter'
    assert model.data[0]['x'] == [0, 1]
    assert model.data[0]['y'] == [2, 3]
    assert model.layout == {'width': 350}
    assert len(model.data_sources) == 1
    assert model.data_sources[0].data == {}
    new_trace = go.Bar(x=[2, 3], y=[4, 5])
    pane.object = {'data': new_trace, 'layout': {'width': 350}}
    assert len(model.data) == 1
    assert model.data[0]['type'] == 'bar'
    assert model.data[0]['x'] == [2, 3]
    assert model.data[0]['y'] == [4, 5]
    assert model.layout == {'width': 350}
    assert len(model.data_sources) == 1
    assert model.data_sources[0].data == {}
    assert pane._models[model.ref['id']][0] is model
    pane._cleanup(model)
    assert pane._models == {}
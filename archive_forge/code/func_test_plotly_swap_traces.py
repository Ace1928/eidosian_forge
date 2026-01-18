import datetime as dt
import pytest
import numpy as np
import pandas as pd
from panel.models.plotly import PlotlyPlot
from panel.pane import PaneBase, Plotly
@plotly_available
def test_plotly_swap_traces(document, comm):
    data_bar = pd.DataFrame({'Count': [1, 2, 3, 4], 'Category': ['A', 'B', 'C', 'D']})
    data_cts = np.random.randn(1000)
    bar_plot = px.bar(x=data_bar['Category'], y=data_bar['Count'])
    dist_plot = ff.create_distplot([data_cts], ['distplot'], bin_size=0.5, show_hist=False, show_rug=False, histnorm='probability')
    plotly = Plotly(bar_plot)
    model = plotly.get_root(document, comm)
    assert len(model.data_sources) == 1
    cds = model.data_sources[0]
    assert (cds.data['x'] == data_bar.Category.values).all()
    assert (cds.data['y'] == data_bar.Count.values).all()
    plotly.object = dist_plot
    assert 'x' not in cds.data
    assert len(cds.data['y'][0]) == 500
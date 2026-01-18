import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
def test_vega_geometry_data(document, comm):
    pane = pn.panel(gdf_example)
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, VegaPlot)
    assert model.data_sources == {}
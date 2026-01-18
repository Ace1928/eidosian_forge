import base64
import os
import sys
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pytest
from bokeh.models import ColorBar
from panel.models.vtk import (
from panel.pane import VTK, PaneBase, VTKVolume
from panel.pane.vtk.vtk import (
@vtk_available
@pytest.mark.skipif(sys.platform == 'win32', reason='cache cleanup fails on windows')
def test_vtk_pane_from_renwin(document, comm):
    renWin = make_render_window()
    pane = VTK(renWin)
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, VTKSynchronizedPlot)
    assert pane._models[model.ref['id']][0] is model
    ctx = pane._contexts[model.id]
    assert len(ctx.dataArrayCache.keys()) == 4
    pane.remove_all_actors()
    assert len(ctx.dataArrayCache.keys()) == 4
    ctx.checkForArraysToRelease(0)
    assert len(ctx.dataArrayCache.keys()) == 0
    pane._cleanup(model)
    assert pane._contexts == {}
    assert pane._models == {}
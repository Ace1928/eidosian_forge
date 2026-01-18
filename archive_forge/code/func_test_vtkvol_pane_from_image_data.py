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
def test_vtkvol_pane_from_image_data(document, comm):
    image_data = make_image_data()
    pane = VTKVolume(image_data)
    from operator import eq
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, VTKVolumePlot)
    assert pane._models[model.ref['id']][0] is model
    assert all([eq(getattr(pane, k), getattr(model, k)) for k in ['slice_i', 'slice_j', 'slice_k']])
    pane._cleanup(model)
    assert pane._models == {}
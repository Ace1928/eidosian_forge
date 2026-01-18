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
@pytest.mark.skip(reason='vtk=9.0.1=no_osmesa not currently available')
def test_vtkjs_pane(document, comm, tmp_path):
    url = 'https://raw.githubusercontent.com/Kitware/vtk-js/master/Data/StanfordDragon.vtkjs'
    pane_from_url = VTK(url)
    model = pane_from_url.get_root(document, comm=comm)
    assert isinstance(model, VTKJSPlot)
    assert pane_from_url._models[model.ref['id']][0] is model
    assert isinstance(model.data, str)
    with BytesIO(base64.b64decode(model.data.encode())) as in_memory:
        with ZipFile(in_memory) as zf:
            filenames = zf.namelist()
            assert len(filenames) == 9
            assert 'StanfordDragon.obj/index.json' in filenames
    tmpfile = os.path.join(*tmp_path.joinpath('export.vtkjs').parts)
    pane_from_url.export_vtkjs(filename=tmpfile)
    with open(tmpfile, 'rb') as file_exported:
        pane_from_url.object = file_exported
    pane_from_file = VTK(tmpfile)
    model_from_file = pane_from_file.get_root(document, comm=comm)
    assert isinstance(pane_from_file, VTKJS)
    assert isinstance(model_from_file, VTKJSPlot)
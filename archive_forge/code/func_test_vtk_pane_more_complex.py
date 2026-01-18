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
@pyvista_available
def test_vtk_pane_more_complex(pyvista_render_window, document, comm, tmp_path):
    pane = VTK(pyvista_render_window)
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, VTKSynchronizedPlot)
    assert pane._models[model.ref['id']][0] is model
    colorbars_infered = pane.construct_colorbars().object
    assert len(colorbars_infered.below) == 2
    assert all((isinstance(cb, ColorBar) for cb in colorbars_infered.below))
    colorbars_in_scene = pane.construct_colorbars(infer=False).object()
    assert len(colorbars_in_scene.below) == 3
    assert all((isinstance(cb, ColorBar) for cb in colorbars_in_scene.below))
    pane.axes = dict(origin=[-5, 5, -2], xticker={'ticks': np.linspace(-5, 5, 5)}, yticker={'ticks': np.linspace(-5, 5, 5)}, zticker={'ticks': np.linspace(-2, 2, 5), 'labels': [''] + [str(int(item)) for item in np.linspace(-2, 2, 5)[1:]]}, fontsize=12, digits=1, grid_opacity=0.5, show_grid=True)
    assert isinstance(model.axes, VTKAxes)
    tmpfile = os.path.join(*tmp_path.joinpath('scene').parts)
    exported_file = pane.export_scene(filename=tmpfile)
    assert exported_file.endswith('.synch')
    imported_scene = VTK.import_scene(filename=exported_file)
    assert isinstance(imported_scene, VTKRenderWindowSynchronized)
    pane._cleanup(model)
    assert pane._contexts == {}
    assert pane._models == {}
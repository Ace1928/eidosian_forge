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
def test_vtk_sync_helpers(document, comm):
    renWin1 = make_render_window()
    renWin2 = make_render_window()
    pane1 = VTK(renWin1)
    pane2 = VTK(renWin2)
    assert isinstance(pane1, VTKRenderWindowSynchronized)
    assert isinstance(pane2, VTKRenderWindowSynchronized)
    model1 = pane1.get_root(document, comm=comm)
    model2 = pane2.get_root(document, comm=comm)
    assert isinstance(model1, VTKSynchronizedPlot)
    assert isinstance(model2, VTKSynchronizedPlot)
    assert len(pane1.actors) == 2
    assert len(pane2.actors) == 2
    assert pane1.actors[0] is not pane2.actors[0]
    pane1.add_actors(pane2.actors)
    assert len(pane1.actors) == 4
    assert pane1.actors[3] is pane2.actors[1]
    save_actor = pane1.actors[0]
    pane1.remove_actors([pane1.actors[0]])
    assert pane1.actors[2] is pane2.actors[1]
    pane1.add_actors([save_actor])
    assert len(pane1.actors) == 4
    pane1.remove_all_actors()
    assert len(pane1.actors) == 0
    save_vtk_camera2 = pane2.vtk_camera
    assert pane1.vtk_camera is not save_vtk_camera2
    pane1.link_camera(pane2)
    assert pane1.vtk_camera is save_vtk_camera2
    pane2.unlink_camera()
    assert pane2.vtk_camera is not save_vtk_camera2
    pane1.set_background(0, 0, 0)
    assert list(renWin1.GetRenderers())[0].GetBackground() == (0, 0, 0)
    pane1._cleanup(model1)
    pane2._cleanup(model2)
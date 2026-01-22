from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def add_picker(self):
    """Create a cell picker."""
    picker = vtk.vtkCellPicker()
    source = vtk.vtkVectorText()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    follower = vtk.vtkFollower()
    follower.SetMapper(mapper)
    follower.GetProperty().SetColor((0, 0, 0))
    follower.SetScale(0.2)
    self.ren.AddActor(follower)
    follower.SetCamera(self.ren.GetActiveCamera())
    follower.VisibilityOff()

    def annotate_pick(obj, event):
        if picker.GetCellId() < 0:
            follower.VisibilityOff()
        else:
            pick_pos = picker.GetPickPosition()
            mapper = picker.GetMapper()
            if mapper in self.mapper_map:
                site = self.mapper_map[mapper]
                output = [site.species_string, 'Frac. coords: ' + ' '.join((f'{c:.4f}' for c in site.frac_coords))]
                source.SetText('\n'.join(output))
                follower.SetPosition(pick_pos)
                follower.VisibilityOn()
    picker.AddObserver('EndPickEvent', annotate_pick)
    self.picker = picker
    self.iren.SetPicker(picker)
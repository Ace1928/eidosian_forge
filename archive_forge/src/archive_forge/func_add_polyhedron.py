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
def add_polyhedron(self, neighbors, center, color, opacity=1.0, draw_edges=False, edges_color=(0.0, 0.0, 0.0), edges_linewidth=2):
    """
        Adds a polyhedron.

        Args:
            neighbors: Neighbors of the polyhedron (the vertices).
            center: The atom in the center of the polyhedron.
            color: Color for text as RGB.
            opacity: Opacity of the polyhedron
            draw_edges: If set to True, the a line will be drawn at each edge
            edges_color: Color of the line for the edges
            edges_linewidth: Width of the line drawn for the edges
        """
    points = vtk.vtkPoints()
    conv = vtk.vtkConvexPointSet()
    for idx, neighbor in enumerate(neighbors):
        x, y, z = neighbor.coords
        points.InsertPoint(idx, x, y, z)
        conv.GetPointIds().InsertId(idx, idx)
    grid = vtk.vtkUnstructuredGrid()
    grid.Allocate(1, 1)
    grid.InsertNextCell(conv.GetCellType(), conv.GetPointIds())
    grid.SetPoints(points)
    dsm = vtk.vtkDataSetMapper()
    poly_sites = [center]
    poly_sites.extend(neighbors)
    self.mapper_map[dsm] = poly_sites
    if vtk.VTK_MAJOR_VERSION <= 5:
        dsm.SetInputConnection(grid.GetProducerPort())
    else:
        dsm.SetInputData(grid)
    ac = vtk.vtkActor()
    ac.SetMapper(dsm)
    ac.GetProperty().SetOpacity(opacity)
    if color == 'element':
        max_occu = 0.0
        for specie, occu in center.species.items():
            if occu > max_occu:
                max_specie = specie
                max_occu = occu
        color = [i / 255 for i in self.el_color_mapping[max_specie.symbol]]
        ac.GetProperty().SetColor(color)
    else:
        ac.GetProperty().SetColor(color)
    if draw_edges:
        ac.GetProperty().SetEdgeColor(edges_color)
        ac.GetProperty().SetLineWidth(edges_linewidth)
        ac.GetProperty().EdgeVisibilityOn()
    self.ren.AddActor(ac)
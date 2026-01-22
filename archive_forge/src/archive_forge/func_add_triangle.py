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
def add_triangle(self, neighbors, color, center=None, opacity=0.4, draw_edges=False, edges_color=(0.0, 0.0, 0.0), edges_linewidth=2):
    """
        Adds a triangular surface between three atoms.

        Args:
            neighbors: Atoms between which a triangle will be drawn.
            color: Color for triangle as RGB.
            center: The "central atom" of the triangle
            opacity: opacity of the triangle
            draw_edges: If set to True, the a line will be  drawn at each edge
            edges_color: Color of the line for the edges
            edges_linewidth: Width of the line drawn for the edges
        """
    points = vtk.vtkPoints()
    triangle = vtk.vtkTriangle()
    for ii in range(3):
        points.InsertNextPoint(neighbors[ii].x, neighbors[ii].y, neighbors[ii].z)
        triangle.GetPointIds().SetId(ii, ii)
    triangles = vtk.vtkCellArray()
    triangles.InsertNextCell(triangle)
    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(trianglePolyData)
    ac = vtk.vtkActor()
    ac.SetMapper(mapper)
    ac.GetProperty().SetOpacity(opacity)
    if color == 'element':
        if center is None:
            raise ValueError('Color should be chosen according to the central atom, and central atom is not provided')
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
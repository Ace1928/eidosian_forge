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
def add_faces(self, faces, color, opacity=0.35):
    """
        Adding face of polygon.

        Args:
            faces (): Coordinates of the faces.
            color (): Color.
            opacity (float): Opacity
        """
    for face in faces:
        if len(face) == 3:
            points = vtk.vtkPoints()
            triangle = vtk.vtkTriangle()
            for ii in range(3):
                points.InsertNextPoint(face[ii][0], face[ii][1], face[ii][2])
                triangle.GetPointIds().SetId(ii, ii)
            triangles = vtk.vtkCellArray()
            triangles.InsertNextCell(triangle)
            trianglePolyData = vtk.vtkPolyData()
            trianglePolyData.SetPoints(points)
            trianglePolyData.SetPolys(triangles)
            mapper = vtk.vtkPolyDataMapper()
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInputConnection(trianglePolyData.GetProducerPort())
            else:
                mapper.SetInputData(trianglePolyData)
            ac = vtk.vtkActor()
            ac.SetMapper(mapper)
            ac.GetProperty().SetOpacity(opacity)
            ac.GetProperty().SetColor(color)
            self.ren.AddActor(ac)
        elif len(face) > 3:
            center = np.zeros(3, float)
            for site in face:
                center += site
            center /= np.float64(len(face))
            for ii, f in enumerate(face, start=1):
                points = vtk.vtkPoints()
                triangle = vtk.vtkTriangle()
                points.InsertNextPoint(f[0], f[1], f[2])
                ii2 = np.mod(ii, len(face))
                points.InsertNextPoint(face[ii2][0], face[ii2][1], face[ii2][2])
                points.InsertNextPoint(center[0], center[1], center[2])
                for jj in range(3):
                    triangle.GetPointIds().SetId(jj, jj)
                triangles = vtk.vtkCellArray()
                triangles.InsertNextCell(triangle)
                trianglePolyData = vtk.vtkPolyData()
                trianglePolyData.SetPoints(points)
                trianglePolyData.SetPolys(triangles)
                mapper = vtk.vtkPolyDataMapper()
                if vtk.VTK_MAJOR_VERSION <= 5:
                    mapper.SetInputConnection(trianglePolyData.GetProducerPort())
                else:
                    mapper.SetInputData(trianglePolyData)
                ac = vtk.vtkActor()
                ac.SetMapper(mapper)
                ac.GetProperty().SetOpacity(opacity)
                ac.GetProperty().SetColor(color)
                self.ren.AddActor(ac)
        else:
            raise ValueError('Number of points for a face should be >= 3')
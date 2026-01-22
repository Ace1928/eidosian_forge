import os.path as op
import numpy as np
from numpy import linalg as nla
from .. import logging
from ..interfaces.base import (
from ..interfaces.vtkbase import tvtk
from ..interfaces import vtkbase as VTKInfo
class ComputeMeshWarp(TVTKBaseInterface):
    """
    Calculates a the vertex-wise warping to get surface2 from surface1.
    It also reports the average distance of vertices, using the norm specified
    as input.

    .. warning:

      A point-to-point correspondence between surfaces is required


    Example::

        import nipype.algorithms.mesh as m
        dist = m.ComputeMeshWarp()
        dist.inputs.surface1 = 'surf1.vtk'
        dist.inputs.surface2 = 'surf2.vtk'
        res = dist.run()

    """
    input_spec = ComputeMeshWarpInputSpec
    output_spec = ComputeMeshWarpOutputSpec

    def _triangle_area(self, A, B, C):
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        ABxAC = nla.norm(A - B) * nla.norm(A - C)
        prod = np.dot(B - A, C - A)
        angle = np.arccos(prod / ABxAC)
        area = 0.5 * ABxAC * np.sin(angle)
        return area

    def _run_interface(self, runtime):
        r1 = tvtk.PolyDataReader(file_name=self.inputs.surface1)
        r2 = tvtk.PolyDataReader(file_name=self.inputs.surface2)
        vtk1 = VTKInfo.vtk_output(r1)
        vtk2 = VTKInfo.vtk_output(r2)
        r1.update()
        r2.update()
        assert len(vtk1.points) == len(vtk2.points)
        points1 = np.array(vtk1.points)
        points2 = np.array(vtk2.points)
        diff = points2 - points1
        weights = np.ones(len(diff))
        try:
            errvector = nla.norm(diff, axis=1)
        except TypeError:
            errvector = np.apply_along_axis(nla.norm, 1, diff)
        if self.inputs.metric == 'sqeuclidean':
            errvector **= 2
        if self.inputs.weighting == 'area':
            faces = vtk1.polys.to_array().reshape(-1, 4).astype(int)[:, 1:]
            for i, p1 in enumerate(points2):
                w = 0.0
                point_faces = faces[(faces[:, :] == i).any(axis=1)]
                for idset in point_faces:
                    fp1 = points1[int(idset[0])]
                    fp2 = points1[int(idset[1])]
                    fp3 = points1[int(idset[2])]
                    w += self._triangle_area(fp1, fp2, fp3)
                weights[i] = w
        result = np.vstack([errvector, weights])
        np.save(op.abspath(self.inputs.out_file), result.transpose())
        out_mesh = tvtk.PolyData()
        out_mesh.points = vtk1.points
        out_mesh.polys = vtk1.polys
        out_mesh.point_data.vectors = diff
        out_mesh.point_data.vectors.name = 'warpings'
        writer = tvtk.PolyDataWriter(file_name=op.abspath(self.inputs.out_warp))
        VTKInfo.configure_input_data(writer, out_mesh)
        writer.write()
        self._distance = np.average(errvector, weights=weights)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        outputs['out_warp'] = op.abspath(self.inputs.out_warp)
        outputs['distance'] = self._distance
        return outputs
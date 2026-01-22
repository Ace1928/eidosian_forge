import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class VtkStreamlines(StdOutCommandLine):
    """
    Use vtkstreamlines to convert raw or voxel format streamlines to VTK polydata

    Examples
    --------

    >>> import nipype.interfaces.camino as cmon
    >>> vtk = cmon.VtkStreamlines()
    >>> vtk.inputs.in_file = 'tract_data.Bfloat'
    >>> vtk.inputs.voxeldims = [1,1,1]
    >>> vtk.run()                  # doctest: +SKIP
    """
    _cmd = 'vtkstreamlines'
    input_spec = VtkStreamlinesInputSpec
    output_spec = VtkStreamlinesOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['vtk'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '.vtk'
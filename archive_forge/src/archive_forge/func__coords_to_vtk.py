import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
def _coords_to_vtk(self, points, out_file):
    from ..vtkbase import tvtk
    from ...interfaces import vtkbase as VTKInfo
    if VTKInfo.no_tvtk():
        raise ImportError('TVTK is required and tvtk package was not found')
    reader = tvtk.PolyDataReader(file_name=self.inputs.in_file)
    reader.update()
    mesh = VTKInfo.vtk_output(reader)
    mesh.points = points
    writer = tvtk.PolyDataWriter(file_name=out_file)
    VTKInfo.configure_input_data(writer, mesh)
    writer.write()
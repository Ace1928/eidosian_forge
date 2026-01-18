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
def test_vtkvol_serialization_coherence(document, comm):
    from vtk.util import numpy_support
    data_matrix = np.zeros([50, 75, 100], dtype=np.uint8)
    data_matrix[0:35, 0:35, 0:35] = 50
    data_matrix[25:50, 25:55, 25:55] = 100
    data_matrix[45:50, 45:74, 45:100] = 150
    origin = (0, 10, 20)
    spacing = (3, 2, 1)
    data_matrix_c = np.ascontiguousarray(data_matrix)
    data_matrix_f = np.asfortranarray(data_matrix)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(*data_matrix.shape)
    image_data.SetOrigin(*origin)
    image_data.SetSpacing(*spacing)
    vtk_arr = numpy_support.numpy_to_vtk(data_matrix.ravel(order='F'))
    image_data.GetPointData().SetScalars(vtk_arr)
    p_c = VTKVolume(data_matrix_c, origin=origin, spacing=spacing)
    p_f = VTKVolume(data_matrix_f, origin=origin, spacing=spacing)
    p_id = VTKVolume(image_data)
    assert p_c._sub_spacing == p_f._sub_spacing == p_id._sub_spacing == spacing
    vd_c = p_c._get_volume_data()
    vd_f = p_f._get_volume_data()
    vd_id = p_id._get_volume_data()
    data_decoded = np.frombuffer(base64.b64decode(vd_c['buffer']), dtype=vd_c['dtype']).reshape(vd_c['dims'], order='F')
    assert np.alltrue(data_decoded == data_matrix)
    assert vd_id == vd_c == vd_f
    p_c_ds = VTKVolume(data_matrix_c, origin=origin, spacing=spacing, max_data_size=0.1)
    p_f_ds = VTKVolume(data_matrix_f, origin=origin, spacing=spacing, max_data_size=0.1)
    p_id_ds = VTKVolume(image_data, max_data_size=0.1)
    assert p_c_ds._sub_spacing == p_f_ds._sub_spacing == p_id_ds._sub_spacing != spacing
    vd_c_ds = p_c_ds._get_volume_data()
    vd_f_ds = p_f_ds._get_volume_data()
    vd_id_ds = p_id_ds._get_volume_data()
    assert vd_id_ds == vd_c_ds == vd_f_ds
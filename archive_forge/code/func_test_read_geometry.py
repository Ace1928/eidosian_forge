import io
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory
@needs_nibabel_data('nitest-cifti2')
def test_read_geometry():
    img = ci.Cifti2Image.from_filename(DATA_FILE6)
    geometry_mapping = img.header.matrix.get_index_map(1)
    expected_geometry = [('CIFTI_STRUCTURE_CORTEX_LEFT', 29696, 0, 32491), ('CIFTI_STRUCTURE_CORTEX_RIGHT', 29716, 0, 32491), ('CIFTI_STRUCTURE_ACCUMBENS_LEFT', 135, [49, 66, 28], [48, 72, 35]), ('CIFTI_STRUCTURE_ACCUMBENS_RIGHT', 140, [40, 66, 29], [43, 66, 36]), ('CIFTI_STRUCTURE_AMYGDALA_LEFT', 315, [55, 61, 21], [56, 58, 31]), ('CIFTI_STRUCTURE_AMYGDALA_RIGHT', 332, [34, 62, 20], [36, 61, 31]), ('CIFTI_STRUCTURE_BRAIN_STEM', 3472, [42, 41, 0], [46, 50, 36]), ('CIFTI_STRUCTURE_CAUDATE_LEFT', 728, [50, 72, 32], [53, 60, 49]), ('CIFTI_STRUCTURE_CAUDATE_RIGHT', 755, [40, 68, 33], [37, 62, 49]), ('CIFTI_STRUCTURE_CEREBELLUM_LEFT', 8709, [49, 35, 4], [46, 37, 37]), ('CIFTI_STRUCTURE_CEREBELLUM_RIGHT', 9144, [38, 35, 4], [44, 38, 36]), ('CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT', 706, [52, 53, 26], [56, 49, 35]), ('CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT', 712, [39, 54, 26], [35, 49, 36]), ('CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT', 764, [55, 60, 21], [54, 44, 39]), ('CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT', 795, [33, 60, 21], [38, 45, 39]), ('CIFTI_STRUCTURE_PALLIDUM_LEFT', 297, [56, 59, 32], [55, 61, 39]), ('CIFTI_STRUCTURE_PALLIDUM_RIGHT', 260, [36, 62, 32], [35, 62, 39]), ('CIFTI_STRUCTURE_PUTAMEN_LEFT', 1060, [51, 66, 28], [58, 64, 43]), ('CIFTI_STRUCTURE_PUTAMEN_RIGHT', 1010, [34, 66, 29], [31, 62, 43]), ('CIFTI_STRUCTURE_THALAMUS_LEFT', 1288, [55, 47, 33], [52, 53, 46]), ('CIFTI_STRUCTURE_THALAMUS_RIGHT', 1248, [32, 47, 34], [38, 55, 46])]
    current_index = 0
    for from_file, expected in zip(geometry_mapping.brain_models, expected_geometry):
        assert from_file.model_type in ('CIFTI_MODEL_TYPE_SURFACE', 'CIFTI_MODEL_TYPE_VOXELS')
        assert from_file.brain_structure == expected[0]
        assert from_file.index_offset == current_index
        assert from_file.index_count == expected[1]
        current_index += from_file.index_count
        if from_file.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
            assert from_file.voxel_indices_ijk is None
            assert len(from_file.vertex_indices) == expected[1]
            assert from_file.vertex_indices[0] == expected[2]
            assert from_file.vertex_indices[-1] == expected[3]
            assert from_file.surface_number_of_vertices == 32492
        else:
            assert from_file.vertex_indices is None
            assert from_file.surface_number_of_vertices is None
            assert len(from_file.voxel_indices_ijk) == expected[1]
            assert from_file.voxel_indices_ijk[0] == expected[2]
            assert from_file.voxel_indices_ijk[-1] == expected[3]
    assert current_index == img.shape[1]
    expected_affine = [[-2, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]]
    expected_dimensions = (91, 109, 91)
    assert np.array_equal(geometry_mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix, expected_affine)
    assert geometry_mapping.volume.volume_dimensions == expected_dimensions